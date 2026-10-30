import logging
import asyncio
import time
from typing import List

from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery

from agentweave.agents.base import BaseAgent
from agentweave.core.modules_rag import ModelRegistry
from agentweave.scheduler.state import(
    RequestState,
    RoundTask,
    GLOBAL_REQUEST_STATES_QUEUE,
    GLOBAL_FINISHED_REQUEST_STATES
)
from agentweave.utils.split import dispatch_text_into_nodes

logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):

    def get_initial_state_config(self, doc: str, config: dict) -> dict:
        defaults = {
            "top_k": 8,
            "rerank_top_k": 5,
            "text_size": 1024,
        }
        defaults.update(config)
        return defaults

    async def process_batch(self, tasks: List[RoundTask], states: List[RequestState]):
        # --- 1. Pre-processing: Lazy Initialization ---
        states_needing_init = [s for s in states if s.payload["rewritten_query"] is None]

        if states_needing_init:
            original_queries = [s.original_query for s in states_needing_init]
            rewritten_list = await ModelRegistry.llm.async_generate_rewrite_query(
                original_queries
            )
            q_embeddings = await ModelRegistry.embed.async_encode(rewritten_list)

            for s, rw_q, q_emb in zip(
                states_needing_init, rewritten_list, q_embeddings
            ):
                s.payload["rewritten_query"] = rw_q
                s.payload["query_embedding"] = q_emb

        # --- 2. Load & Split & Embed Nodes ---
        all_nodes = {}
        all_node_texts = []
        node_map_list = []

        for t in tasks:
            nodes = dispatch_text_into_nodes(
                t.doc_chunk_text, t.request_id
            )
            all_nodes[t.request_id] = nodes
            if nodes:
                for node in nodes:
                    all_node_texts.append(node.get_content()+"\n")
                    node_map_list.append(node)

        # Batch Embed Documents
        if all_node_texts:
            doc_embeddings = await ModelRegistry.embed.async_encode(all_node_texts)
            assert len(doc_embeddings) == len(node_map_list)
            for node, emb in zip(node_map_list, doc_embeddings):
                node.embedding = emb

        # --- 3. Retrieval & Rerank Prep ---
        rerank_inputs = []  # List[Tuple[query, doc_text]]
        retrieved_nodes_map = {}  # req_id -> List[NodeWithScore]

        for i, (task, state) in enumerate(zip(tasks, states)):
            nodes = all_nodes.get(state.req_id, [])
            if not nodes or not nodes[0].embedding:
                retrieved_nodes_map[state.req_id] = []
                continue

            vector_store = SimpleVectorStore()
            vector_store.add(nodes)

            query_res = vector_store.query(
                VectorStoreQuery(
                    query_embedding=state.payload["query_embedding"],
                    similarity_top_k=state.config["top_k"],
                )
            )

            node_with_score = []
            nodes_by_id = {node.node_id: node for node in nodes}
            for node_id, sim in zip(query_res.ids, query_res.similarities):
                node = nodes_by_id[node_id]
                node_with_score.append(NodeWithScore(node=node, score=sim))
                rerank_inputs.append([state.payload["rewritten_query"], node.get_content()])

            retrieved_nodes_map[state.req_id] = node_with_score

        # --- 4. Batch Rerank ---
        if rerank_inputs:
            instruction = "Given a query, retrieve the most relevant document that answers the query."
            rerank_scores = await ModelRegistry.reranker.async_rerank(
                rerank_inputs, instruction
            )

            score_idx = 0
            for state in states:
                nodes = retrieved_nodes_map.get(state.req_id, [])
                for i in range(len(nodes)):
                    nodes[i].score = rerank_scores[score_idx + i]
                score_idx += len(nodes)
            assert score_idx == len(rerank_scores), "Rerank scores count mismatch."

        # --- 5. Construct Prompts (Core Logic) ---
        prompts_for_llm = []
        sampling_params = []

        system_prompt = (
            "You are a helpful Q&A assistant. "
            "Your task is to answer the user's question based on the provided context. "
            "Only output the answer text. "
            "If the context information does not contain relevant content, respond with 'No relevant information found.' instead of making up an answer.\n"
        )
        user_prompt_template = (
            "Given the context information and not prior knowledge, please answer the query.\n"
            "Query: {query_str}\n"
            "Context information is provided below:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
        )
        postfix = "\n---------------------\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        for task, state in zip(tasks, states):
            nodes = retrieved_nodes_map.get(state.req_id, [])
            nodes.sort(key=lambda x: x.score, reverse=True)
            top_nodes = nodes[: state.config["rerank_top_k"]]

            context_str = "\n\n".join([n.node.get_content() for n in top_nodes])
            if not context_str:
                context_str = "No relevant context."
                logger.info(f"[Req {state.req_id}] No relevant context.")

            if state.payload["prompt_history"]:
                if task.is_last_round:
                    # Case A-1: Decode
                    final_prompt = (
                        state.payload["prompt_history"] + "\n\n" + context_str + "\n" + postfix
                    )
                    sampling = {"max_new_tokens": 1024}
                else:
                    # Case A-2: Prefill
                    final_prompt = state.payload["prompt_history"] + "\n\n" + context_str
                    sampling = {"max_new_tokens": 0}
                    state.payload["prompt_history"] = final_prompt
            else:
                user_prompt = user_prompt_template.format(
                    context_str=context_str, query_str=state.payload["rewritten_query"]
                )
                chat_prompt = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                base_prompt = ModelRegistry.llm.tokenizer.apply_chat_template(
                    chat_prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ).replace(postfix, "")

                if task.is_last_round:
                    final_prompt = base_prompt + postfix
                    sampling = {"max_new_tokens": 1024}
                else:
                    final_prompt = base_prompt
                    sampling = {"max_new_tokens": 0}
                    state.payload["prompt_history"] = final_prompt

            prompts_for_llm.append(final_prompt)
            sampling_params.append(sampling)

        # --- 6. Batch LLM ---
        logger.info(
            f"Running LLM Batch: {len(prompts_for_llm)} prompts. Types: {['Decode' if p['max_new_tokens'] > 0 else 'Prefill' for p in sampling_params]}"
        )
        responses = await ModelRegistry.llm.async_generate(
            prompts_for_llm, sampling_params
        )

        # --- 7. Update State & Next Round ---
        for task, state, response in zip(tasks, states, responses):
            state.current_round_idx += 1
            if task.is_last_round:
                state.round_answers.append(response)
                state.status = "finished"
                state.finished_at = time.perf_counter()
                GLOBAL_FINISHED_REQUEST_STATES[state.req_id] = state
                logger.info(
                    f"[Req {state.req_id}] ✅ Finished. Metrics: {state.metrics}. Final Answer: {response[:50]}...\n"
                )

            else:
                state.round_answers.append("(Prefill)")
                state.priority -= 1
                await GLOBAL_REQUEST_STATES_QUEUE.put(state)
                logger.info(
                    f"[Request {state.req_id}] Round {task.round_idx} done."
                )
