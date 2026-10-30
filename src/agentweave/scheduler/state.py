from __future__ import annotations
import time
import asyncio
from typing import List, Dict, Any, Optional
from functools import total_ordering

from agentweave.utils.config import app_config

GLOBAL_REQUEST_STATES_QUEUE: asyncio.PriorityQueue[RequestState] = (
    asyncio.PriorityQueue()
)
GLOBAL_FINISHED_REQUEST_STATES: Dict[
    str, RequestState
] = {}
GLOBAL_REQ_CNT = 0


class RoundTask:

    def __init__(
        self,
        request_id: str,
        round_idx: int,
        is_last_round: bool,
        doc_chunk_text: str = "",
    ):
        self.request_id = request_id
        self.round_idx = round_idx
        self.is_last_round = is_last_round
        self.doc_chunk_text = doc_chunk_text





@total_ordering
class RequestState:

    def __init__(
        self,
        req_id: str = "",
        query: str = "",
        payload: Dict[str, Any] = {},
        # doc_text: List[str] = [""],
        # doc_token: List = [],
        config: Dict[str, Any] = {},
        priority: int = 1000,
    ):
        self.priority = priority
        self.req_id = req_id
        self.original_query = query
        self.config = config

        ################# RAG #################
        # self.payload["doc_text"]
        # self.payload["doc_token"]
        #
        # self.payload["rewritten_query"]: Optional[str] = None
        # self.payload["query_embedding"]: Optional[List[float]] = None
        #
        # self.payload["prompt_history"]
        #######################################

        ################# RPG #################
        # self.payload["recaption"]: Optional[str] = None
        # self.payload["subprompts"]: Optional[List[str]] = None
        #
        # self.payload["planning_history"]

        # self.payload["decomposition_size"]
        # self.payload["remaining_n"]
        #######################################

        ################# KB_RAG #################
        # self.payload["remaining_k"]
        # self.payload["decomposition_size"]
        #
        # self.payload["rewritten_query"]: Optional[str] = None
        # self.payload["query_embedding"]: Optional[List[float]] = None
        #
        # self.payload["prompt_history"]
        #######################################
        self.payload = payload

        self.current_round_idx = 0

        self.round_answers: List[str] = []
        self.status: str = "pending"

        self.created_at = time.perf_counter()
        self.processing_start_at = None
        self.finished_at = None

    def is_last_round(self):
        agent_type = app_config.args.get("agent_type", "")

        # KB-RAG
        if agent_type == "rag" and "doc_text" in self.payload:
            return len(self.payload["doc_text"]) == 0
        elif agent_type == "kb_rag" and "remaining_k" in self.payload:
            return self.payload["remaining_k"] <= 0
        # RPG
        elif agent_type == "rpg" and "remaining_n" in self.payload:
            return self.payload["remaining_n"] == 0
        # Deep Search
        elif agent_type == "deep_search":
            num_subagents = self.config.get("num_subagents")
            end_idx = self.payload.get("end_idx", 0)
            return end_idx >= num_subagents
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")

    @property
    def metrics(self) -> Dict[str, float]:
        if self.finished_at is None:
            return {}

        total_latency = self.finished_at - self.created_at

        queue_latency = (
            (self.processing_start_at - self.created_at)
            if self.processing_start_at
            else 0.0
        )

        exec_latency = total_latency - queue_latency

        return {
            "total_latency": round(total_latency, 4),
            "queue_latency": round(queue_latency, 4),
            "exec_latency": round(exec_latency, 4),
        }

    def __lt__(self, other):
        if not isinstance(other, RequestState):
            return NotImplemented
        return self.priority < other.priority

    def __eq__(self, other):
        if not isinstance(other, RequestState):
            return NotImplemented
        return self.priority == other.priority

    def __repr__(self):
        return None
