import asyncio
import logging
import json
import time
from typing import List, Optional, Dict, Any

from agentweave.agents.base import BaseAgent
from agentweave.core.modules_deep_search import ModelRegistry
from agentweave.scheduler.state import (
    RequestState,
    RoundTask,
    GLOBAL_REQUEST_STATES_QUEUE,
    GLOBAL_FINISHED_REQUEST_STATES,
)
from agentweave.utils.config import app_config
from agentweave.utils.web_search import get_search_client

logger = logging.getLogger(__name__)

LEAD_RESEARCHER_SYSTEM_PROMPT = """You are the Lead Researcher, responsible for breaking down complex queries into research questions.

Your task is to:
1. Analyze the user's query
2. Generate all research questions at once
3. Each question should explore a different aspect of the topic

Output a JSON array with all questions.
"""

BATCH_SUBTASK_GENERATION_TEMPLATE = """
User Query: {query}

Generate at least {num_questions} research questions for this query based on the difficulty level.

Output a JSON array (no wrapper object):
[
    {{
        "question": "specific research question 1",
        "search_query": "query for web search 1"
    }},
    {{
        "question": "specific research question 2",
        "search_query": "query for web search 2"
    }}
]

Output the JSON array only, no additional text.
"""

SUBAGENT_SYSTEM_PROMPT = """You are a Research Subagent evaluator. Evaluate the quality and relevance of search results.

Output format (JSON):
{
    "quality": "good" | "partial" | "poor"
}

Quality criteria:
- "good": Results directly answer the research question with relevant, authoritative sources
- "partial": Some relevant information but incomplete or mixed quality
- "poor": Irrelevant, spam, or no useful information found

Output JSON only, no explanation.
"""

SUBAGENT_USER_TEMPLATE = """
Research Question: {question}
Search Query: {search_query}

{search_context}

Evaluate the search results quality above. Output JSON only.
"""

POSTFIX = "\n---------------------\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


# ============================================================================
# Deep Search Agent
# ============================================================================


class DeepSearchAgent(BaseAgent):

    def get_initial_state_config(self, doc: str, config: dict) -> dict:
        dr_config = app_config.args["deep_search"]
        defaults = {
            "num_subagents": 8,
            "search_top_k": dr_config.get("search_top_k", 10),
            "enable_citations": True,
        }
        defaults.update(config)
        return defaults

    async def process_batch(self, tasks: List[RoundTask], states: List[RequestState]):
        if not tasks:
            raise ValueError("no task in process_batch.")

        logger.info(f"[DR] Processing {len(tasks)} tasks")
        t_start = time.perf_counter()

        for task, state in zip(tasks, states):
            total_subtasks = state.config.get("num_subagents")
            start_idx = state.payload.get("start_idx")
            end_idx = state.payload.get("end_idx")
            is_last_round = task.is_last_round

            logger.info(
                f"[DR] Req {state.req_id}: Batch [{start_idx}, {end_idx})/{total_subtasks}, "
                f"is_last={is_last_round}"
            )

            if "synthesis_context" not in state.payload:
                state.payload["plan_context"] = self._build_plan_prompt(state)
                state.payload["synthesis_context"] = self._build_synthesis_base_prompt(state)
                state.payload["subagent_results"] = []

        t0 = time.perf_counter()
        all_questions: List[tuple] = []  # (req_idx, question_idx, question, search_query)

        prompts_for_questions: List[str] = []
        question_counts: List[int] = []
        first_round_indices: List[int] = []

        for req_idx, (task, state) in enumerate(zip(tasks, states)):
            start_idx = state.payload.get("start_idx")
            end_idx = state.payload.get("end_idx")

            if "all_questions" not in state.payload:
                prompts_for_questions.append(state.payload["plan_context"])
                question_counts.append(state.config.get("num_subagents", 8))
                first_round_indices.append(req_idx)
            else:
                saved_questions = state.payload["all_questions"]
                for i in range(start_idx, end_idx):
                    if i < len(saved_questions) and saved_questions[i]:
                        all_questions.append((
                            req_idx,
                            i,
                            saved_questions[i]["question"],
                            saved_questions[i].get("search_query", saved_questions[i]["question"]),
                        ))

        if prompts_for_questions:
            logger.info(f"[DR] Phase 1: Batch generating questions for {len(prompts_for_questions)} requests")
            responses = await ModelRegistry.llm.async_generate(
                prompts_for_questions,
                {"max_new_tokens": 4096, "temperature": 0.7},
            )

            for req_idx, response, total_questions in zip(
                first_round_indices, responses, question_counts
            ):
                state = states[req_idx]
                questions = self._parse_batch_questions(response, total_questions)

                state.payload["all_questions"] = questions

                start_idx = state.payload.get("start_idx")
                end_idx = state.payload.get("end_idx")

                for q_idx, parsed in enumerate(questions[start_idx:end_idx], start_idx):
                    if parsed:
                        all_questions.append((
                            req_idx,
                            q_idx,
                            parsed["question"],
                            parsed.get("search_query", parsed["question"]),
                        ))
                    else:
                        logger.warning(f"[DR] Req {state.req_id}: ❌Failed to parse question {q_idx + 1}")

        t1 = time.perf_counter()
        logger.info(f"[DR] Phase 1 (Generate Questions): {len(all_questions)} questions, time={t1-t0:.2f}s")

        search_results: Dict[tuple, str] = {}
        if all_questions:
            search_results = await self._batch_web_search(all_questions)
        t2 = time.perf_counter()
        logger.info(f"[DR] Phase 2 (Web Search): {len(search_results)} searches, time={t2-t1:.2f}s")

        evaluations: Dict[tuple, str] = {}
        if all_questions:
            evaluations = await self._batch_subagent_evaluate(all_questions, search_results)
        t3 = time.perf_counter()
        logger.info(f"[DR] Phase 3 (Subagent LLM): {len(evaluations)} evaluations, time={t3-t2:.2f}s")

        for req_idx, question_idx, question, _ in all_questions:
            state = states[req_idx]
            search_result = search_results.get((req_idx, question_idx), "No search results.")
            evaluation = evaluations.get((req_idx, question_idx), "No evaluation.")

            state.payload["synthesis_context"] += (
                f"\n\n--- Research Result {question_idx + 1} ---\n"
                f"Question: {question}\n"
                f"Search Results:\n{search_result}\n"
                f"Quality Evaluation: {evaluation}\n"
            )

        prompts_for_llm: List[str] = []
        sampling_params: List[dict] = []

        for task, state in zip(tasks, states):
            is_last_round = task.is_last_round

            if is_last_round:
                final_prompt = (
                    state.payload["synthesis_context"]
                    + "\n\nProvide your final answer (1-5 words):"
                    + POSTFIX
                )
                prompts_for_llm.append(final_prompt)
                sampling_params.append({"max_new_tokens": 128, "temperature": 0.7})
            else:
                prompts_for_llm.append(state.payload["synthesis_context"])
                sampling_params.append({"max_new_tokens": 0, "temperature": 0.0})

        logger.info(
            f"[DR] Phase 4 (Lead Agent LLM): {len(prompts_for_llm)} prompts, "
            f"Types: {['Decode' if p['max_new_tokens'] > 0 else 'Prefill' for p in sampling_params]}"
        )
        responses = await ModelRegistry.llm.async_generate(prompts_for_llm, sampling_params)
        t4 = time.perf_counter()
        logger.info(f"[DR] Phase 4 (Lead Agent LLM): time={t4-t3:.2f}s")

        for task, state, response in zip(tasks, states, responses):
            start_idx = state.payload.get("start_idx")
            end_idx = state.payload.get("end_idx")
            is_last_round = task.is_last_round

            if is_last_round:
                logger.info(f"[DR] Req {state.req_id}: ✅ Done, answer length={len(response)}")
                state.round_answers.append({"answer": response})
                state.status = "finished"
                state.finished_at = time.perf_counter()
                GLOBAL_FINISHED_REQUEST_STATES[state.req_id] = state
            else:
                logger.info(
                    f"[DR] Req {state.req_id}: Prefill completed, length={len(state.payload['synthesis_context'])}"
                )
                state.round_answers.append("(Prefill)")
                state.priority -= 1
                await GLOBAL_REQUEST_STATES_QUEUE.put(state)
                logger.info(
                    f"[DR] Req {state.req_id}: Batch [{start_idx}, {end_idx}) completed, re-queued"
                )

        t_end = time.perf_counter()
        logger.info(f"[DR] Total batch time: {t_end - t_start:.2f}s")

    async def _batch_web_search(
        self, questions: List[tuple]
    ) -> Dict[tuple, str]:
        search_client = get_search_client()

        async def _search(req_idx, question_idx, search_query):
            try:
                result = await search_client.web_search(
                    query=search_query,
                    top_k=10,
                    request_id=str(req_idx + 1),
                    question_idx=question_idx,
                )
                return (req_idx, question_idx, result if result else "No search results available.")
            except Exception as e:
                logger.warning(f"[Search] Req {req_idx + 1} Q{question_idx + 1} ❌failed: {e}")
                return (req_idx, question_idx, "Search failed.")

        tasks = [_search(req_idx, q_idx, search_query) for req_idx, q_idx, _, search_query in questions]
        results = await asyncio.gather(*tasks)

        search_results = {}
        for req_idx, question_idx, result in results:
            search_results[(req_idx, question_idx)] = result

        return search_results

    async def _batch_subagent_evaluate(
        self, questions: List[tuple], search_results: Dict[tuple, str]
    ) -> Dict[tuple, str]:
        prompts = []
        for req_idx, question_idx, question, search_query in questions:
            search_context = search_results.get((req_idx, question_idx), "No search results.")
            user_prompt = SUBAGENT_USER_TEMPLATE.format(
                question=question,
                search_query=search_query,
                search_context=search_context,
            )
            chat_prompt = [
                {"role": "system", "content": SUBAGENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            formatted_prompt = ModelRegistry.small_llm.tokenizer.apply_chat_template(
                chat_prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append(formatted_prompt)

        logger.info(f"[DR] Phase 3: Calling subagent LLM for {len(prompts)} prompts")
        responses = await ModelRegistry.small_llm.async_generate(
            prompts, {"max_new_tokens": 128, "temperature": 0.7}
        )

        evaluations = {}
        for i, (req_idx, question_idx, _, _) in enumerate(questions):
            evaluations[(req_idx, question_idx)] = responses[i]

        return evaluations

    def _parse_batch_questions(self, response: str, expected_count: int) -> List[Optional[Dict[str, str]]]:
        import re

        json_str = None

        if "```json" in response:
            match = re.search(r"```json\s*([\{\[].*?[\}\]])\s*```", response, re.DOTALL)
            if match:
                json_str = match.group(1)
        elif "```" in response:
            match = re.search(r"```\s*([\{\[].*?[\}\]])\s*```", response, re.DOTALL)
            if match:
                json_str = match.group(1)

        if json_str is None:
            start_char = None
            start_idx = -1
            for i, char in enumerate(response):
                if char in '{[':
                    start_char = char
                    start_idx = i
                    break

            if start_idx >= 0:
                end_char = '}' if start_char == '{' else ']'
                bracket_count = 0
                for i, char in enumerate(response[start_idx:], start_idx):
                    if char == start_char:
                        bracket_count += 1
                    elif char == end_char:
                        bracket_count -= 1
                        if bracket_count == 0:
                            json_str = response[start_idx:i + 1]
                            break

        if json_str is None:
            logger.warning(f"[DR] ❌Failed to find JSON in response: {response[:200]}")
            return [None] * expected_count

        logger.info(f"[DR] Extracted JSON (first 500 chars): {json_str[:500]}")

        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                questions_list = data
                logger.info(f"[DR] Parsed JSON as array, questions_list length={len(questions_list)}")
            else:
                questions_list = data.get("questions", [])
                logger.info(f"[DR] Parsed JSON, questions_list length={len(questions_list)}, data keys={list(data.keys())}")

            if questions_list:
                logger.info(f"[DR] First question: {questions_list[0] if questions_list else 'N/A'}")

            result = []
            for item in questions_list[:expected_count]:
                result.append({
                    "question": item.get("question", ""),
                    "search_query": item.get("search_query", item.get("question", "")),
                })

            while len(result) < expected_count:
                result.append(None)

            return result
        except json.JSONDecodeError as e:
            logger.warning(f"[DR] ❌Failed to parse JSON: {e}, response: {response[:200]}")
            return [None] * expected_count

    def _build_plan_prompt(self, state: RequestState) -> str:
        chat_prompt = [
            {"role": "system", "content": LEAD_RESEARCHER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": BATCH_SUBTASK_GENERATION_TEMPLATE.format(
                    query=state.original_query,
                    num_questions=state.config.get("num_subagents"),
                ),
            },
        ]
        return ModelRegistry.llm.tokenizer.apply_chat_template(
            chat_prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _build_synthesis_base_prompt(self, state: RequestState) -> str:
        chat_prompt = [
            {
                "role": "system",
                "content": """You are the Lead Researcher. Answer the user's question directly and concisely.

Based on the search results provided, give a VERY BRIEF final answer (1-5 words maximum when possible). Be direct and factual.""",
            },
            {
                "role": "user",
                "content": f"""Original Query: {state.original_query}

I will provide you with research results from multiple subagents. Each includes search results and a quality evaluation.

--- Research Result 1 ---""",
            },
        ]
        return ModelRegistry.llm.tokenizer.apply_chat_template(
            chat_prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ).replace(POSTFIX, "")

    
    