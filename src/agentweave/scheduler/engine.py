import os
import asyncio
import logging
import time
import json
from datetime import datetime
from functools import lru_cache
from typing import Dict, Optional, List

from agentweave.scheduler.state import (
    RequestState,
    RoundTask,
    GLOBAL_FINISHED_REQUEST_STATES,
    GLOBAL_REQUEST_STATES_QUEUE,
    GLOBAL_REQ_CNT,
)
from agentweave.scheduler.algorithm_rag import decomposition_size_algorithm_rag
from agentweave.scheduler.algorithm_kb_rag import decomposition_size_algorithm_kb_rag
from agentweave.scheduler.algorithm_rpg import decomposition_size_algorithm_rpg
from agentweave.scheduler.algorithm_deep_search import (
    decomposition_size_algorithm_deep_search,
)
from agentweave.utils.config import app_config
from agentweave.agents.base import BaseAgent

logger = logging.getLogger(__name__)


ROUND_CNT = 0


class Scheduler:

    def __init__(self, agent: BaseAgent):
        self.agent = agent
        self.batch_size = app_config.args["batch_size"]
        self.create_task_batches_func = {
            "rag": self.create_task_batches_rag,
            "kb_rag": self.create_task_batches_kb_rag,
            "rpg": self.create_task_batches_rpg,
            "deep_search": self.create_task_batches_deep_search,
        }
        self.create_task_batches = self.create_task_batches_func[
            app_config.args["agent_type"]
        ]

    def create_task_batches_rag(self, states: List[RequestState]):
        batch_tasks = []
        start = time.perf_counter()
        decomposition_sizes = decomposition_size_algorithm_rag(states)
        end = time.perf_counter()
        logger.info(f"algorithm time {end - start}")
        for state, decomposition_size in zip(states, decomposition_sizes):
            doc_chunk_text = state.payload["doc_text"][:decomposition_size]
            state.payload["doc_text"] = state.payload["doc_text"][decomposition_size:]
            state.payload["doc_token"] = state.payload["doc_token"][decomposition_size:]
            state.processing_start_at = (
                time.perf_counter()
                if state.processing_start_at is None
                else state.processing_start_at
            )
            task = RoundTask(
                request_id=state.req_id,
                round_idx=state.current_round_idx,
                is_last_round=state.is_last_round(),
                doc_chunk_text=doc_chunk_text,
            )
            batch_tasks.append(task)
        assert len(batch_tasks) == len(states)
        return batch_tasks


    def create_task_batches_kb_rag(self, states: List[RequestState]):
        batch_tasks = []
        decomposition_sizes = decomposition_size_algorithm_kb_rag(states)
        for state, decomposition_size in zip(states, decomposition_sizes):
            state.payload["decomposition_size"] = decomposition_size
            state.payload["remaining_k"] -= decomposition_size
            # if state.config["top_k"] - state.payload["remaining_k"] <= state.config["rerank_top_k"] and state.payload["remaining_k"] > 0:
            #     state.payload["remaining_k"] = 0
            state.processing_start_at = (
                time.perf_counter()
                if state.processing_start_at is None
                else state.processing_start_at
            )
            task = RoundTask(
                request_id=state.req_id,
                round_idx=state.current_round_idx,
                is_last_round=state.is_last_round(),
                doc_chunk_text="",
            )
            batch_tasks.append(task)
        assert len(batch_tasks) == len(states)
        return batch_tasks

    def create_task_batches_rpg(self, states: List[RequestState]):
        batch_tasks = []
        decomposition_sizes = decomposition_size_algorithm_rpg(states)
        for state, decomposition_size in zip(states, decomposition_sizes):
            state.payload["decomposition_size"] = decomposition_size
            state.payload["remaining_n"] -= decomposition_size
            state.processing_start_at = (
                time.perf_counter()
                if state.processing_start_at is None
                else state.processing_start_at
            )
            task = RoundTask(
                request_id=state.req_id,
                round_idx=state.current_round_idx,
                is_last_round=state.is_last_round(),
                doc_chunk_text="",
            )
            batch_tasks.append(task)
        assert len(batch_tasks) == len(states)
        return batch_tasks


    def create_task_batches_deep_search(self, states: List[RequestState]):
        """Deep Search task batch creation - coordinates multi-round research"""
        decomposition_sizes = decomposition_size_algorithm_deep_search(states)

        batch_tasks = []
        for state, decomposition_size in zip(states, decomposition_sizes):
            state.processing_start_at = (
                time.perf_counter()
                if state.processing_start_at is None
                else state.processing_start_at
            )

            total_subtasks = state.config.get("num_subagents", 8)
            start_idx = state.payload.get("end_idx", 0)
            end_idx = min(start_idx + decomposition_size, total_subtasks)

            state.payload["start_idx"] = start_idx
            state.payload["end_idx"] = end_idx

            logger.debug(
                f"[Deep Search] Req {state.req_id}: Round {state.current_round_idx} "
                f"Subtask range [{start_idx}, {end_idx})/{total_subtasks}"
                f"total rounds {total_subtasks}, decomposition_size={decomposition_size}"
            )

            task = RoundTask(
                request_id=state.req_id,
                round_idx=state.current_round_idx,
                is_last_round=state.is_last_round(),
                doc_chunk_text="",
            )
            batch_tasks.append(task)

        assert len(batch_tasks) == len(states)
        return batch_tasks

    async def start_loop(self):
        global ROUND_CNT
        logger.info(
            f"🚀 Scheduler Worker started with agent: {type(self.agent).__name__}"
        )
        while True:
            states = []
            try:
                req_state = await GLOBAL_REQUEST_STATES_QUEUE.get()
                states.append(req_state)
                if req_state.req_id == "1":
                    # choices=["kb_rag", "deep_search", "rpg"],
                    if app_config.args['agent_type']=='deep_search':
                        await asyncio.sleep(1)
                    else:
                        await asyncio.sleep(0.8)

                for _ in range(self.batch_size - 1):
                    if GLOBAL_REQUEST_STATES_QUEUE.empty():
                        break
                    req_state = GLOBAL_REQUEST_STATES_QUEUE.get_nowait()
                    states.append(req_state)
            except Exception as e:
                logger.error(f"Queue error: {e}")
                await asyncio.sleep(0.01)
                continue

            if states:
                logger.info(f"============== ROUND {ROUND_CNT} ==============")
                try:
                    batch_tasks = self.create_task_batches(states)
                    logger.info(f"Created {len(batch_tasks)} tasks")
                    await self.agent.process_batch(batch_tasks, states)
                except Exception as e:
                    logger.exception(f"Batch processing error: {e}")

            ROUND_CNT += 1


_scheduler_instance: Optional[Scheduler] = None


def init_scheduler(agent: BaseAgent):
    global _scheduler_instance
    _scheduler_instance = Scheduler(agent)


async def run_scheduler_loop():
    if _scheduler_instance:
        await _scheduler_instance.start_loop()
    else:
        logger.error("Scheduler not initialized!")


@lru_cache(maxsize=1)
def get_rag_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(app_config.args["rag"]["embed_path"])


async def submit_request_rag(
    query: str, doc: str, rag_config: Optional[dict] = None
) -> str:
    from agentweave.utils.split import split_text_by_text_size

    config = {"top_k": 8, "rerank_top_k": 5, "text_size": 1024}
    config.update(rag_config or {})
    for key in ("top_k", "rerank_top_k", "text_size"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not doc.strip():
        raise ValueError("rag requires a non-empty doc")

    doc_text = split_text_by_text_size(doc, config["text_size"])
    doc_token = [get_rag_tokenizer().encode(text) for text in doc_text]
    global GLOBAL_REQ_CNT
    GLOBAL_REQ_CNT += 1
    req_id = str(GLOBAL_REQ_CNT)
    state = RequestState(
        req_id=req_id,
        query=query,
        payload={
            "doc_text": doc_text,
            "doc_token": doc_token,
            "rewritten_query": None,
            "query_embedding": None,
            "prompt_history": "",
        },
        config=config,
        priority=1000,
    )
    await GLOBAL_REQUEST_STATES_QUEUE.put(state)
    logger.info(f"[Submit] Req {req_id} accepted. Doc Length: {len(doc)}")
    return req_id


async def submit_request_kb_rag(
    query: str, kb_rag_config: Optional[dict] = None
) -> str:
    if kb_rag_config is None:
        kb_rag_config = {}

    defaults = {"top_k": 8, "rerank_top_k": 5}
    defaults.update(kb_rag_config)
    kb_rag_config = defaults
    for key in ("top_k", "rerank_top_k"):
        value = kb_rag_config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")

    global GLOBAL_REQ_CNT
    GLOBAL_REQ_CNT += 1
    req_id = str(GLOBAL_REQ_CNT)
    payload = {
        "rewritten_query": None,
        "query_embedding": None,
        "prompt_history": "",
        "remaining_k": kb_rag_config["top_k"]
    }
    state = RequestState(
        req_id=req_id,
        query=query,
        payload=payload,
        config=kb_rag_config,
        priority=1000,
    )
    await GLOBAL_REQUEST_STATES_QUEUE.put(state)

    logger.info(f"[Submit] Req {req_id} accepted. Top_k: {kb_rag_config['top_k']}, Rerank_top_k: {kb_rag_config['rerank_top_k']}")

    return req_id


async def submit_request_rpg(query: str, rpg_config: Optional[dict] = None) -> str:
    if rpg_config is None:
        rpg_config = {"n": 3}

    global GLOBAL_REQ_CNT
    GLOBAL_REQ_CNT += 1
    req_id = str(GLOBAL_REQ_CNT)
    payload = {
        "recaption": None,
        "subprompts": None,
        "planning_history": "",
        "remaining_n": rpg_config["n"],
    }
    state = RequestState(
        req_id=req_id,
        query=query,
        payload=payload,
        config=rpg_config,
        priority=1000,
    )
    await GLOBAL_REQUEST_STATES_QUEUE.put(state)
    logger.info(f"[Submit] Req {req_id} accepted. N: {rpg_config.get('n', 3)}")
    return req_id




async def submit_request_deep_search(
    query: str, deep_search_config: Optional[dict] = None
) -> str:
    if deep_search_config is None:
        deep_search_config = {}

    if _scheduler_instance:
        deep_search_config = _scheduler_instance.agent.get_initial_state_config(
            "", deep_search_config
        )

    global GLOBAL_REQ_CNT
    GLOBAL_REQ_CNT += 1
    req_id = str(GLOBAL_REQ_CNT)

    payload = {
        "start_idx": 0,
        "end_idx": 0,
    }

    state = RequestState(
        req_id=req_id,
        query=query,
        payload=payload,
        config=deep_search_config,
        priority=1000,
    )
    await GLOBAL_REQUEST_STATES_QUEUE.put(state)
    logger.info(
        f"[Submit] Req {req_id} accepted. Num subagents: {deep_search_config.get('num_subagents', 3)}"
    )
    return req_id


def get_request_state(req_id: str) -> Optional[RequestState]:
    return GLOBAL_FINISHED_REQUEST_STATES.get(req_id)


def analyze_performance():
    performance_data = []
    for req_id, state in GLOBAL_FINISHED_REQUEST_STATES.items():
        answer = state.round_answers[-1] if state.round_answers else ""
        if isinstance(answer, dict):
            answer = answer.get("answer", "")

        data_row = {
            "req_id": req_id,
            "created_at": state.created_at if state.created_at else None,
            "processing_start_at": state.processing_start_at
            if state.processing_start_at
            else None,
            "finished_at": state.finished_at if state.finished_at else None,
            "query": state.original_query if state.original_query else None,
            "answer": answer,
        }

        performance_data.append(data_row)

    save_dir = os.getenv("PERFORMANCE_SAVE_DIR","")
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"agentweave_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{app_config.args['agent_type']}.json"
    save_path = os.path.join(save_dir, file_name)

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(performance_data, f, ensure_ascii=False, indent=4)
        logger.info(f"✅ Performance analysis results saved to: {save_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save JSON: {str(e)}")
