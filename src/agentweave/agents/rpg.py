import logging
import asyncio
import time
import re
from typing import List, Optional

from agentweave.agents.base import BaseAgent
from agentweave.core.modules_rpg import ModelRegistry
from agentweave.scheduler.state import(
    RequestState, 
    RoundTask, 
    GLOBAL_REQUEST_STATES_QUEUE, 
    GLOBAL_FINISHED_REQUEST_STATES
)
from agentweave.utils.config import app_config

logger = logging.getLogger(__name__)


class RPGAgent(BaseAgent):

    def get_initial_state_config(self, doc: str, config: dict) -> dict:
        defaults = {
            "n": 5,
        }
        defaults.update(config)
        return defaults
    
    def parse_llm_output(self, responses: List[str]) -> List[List[str]]:
        batch_results = []
        marker = "Recaptioned subprompts:"
        for response in responses:
            current_subprompts = []
            if marker in response:
                content_section = response.split(marker)[-1]
            else:
                content_section = response
            raw_chunks = content_section.split("=====")
            for chunk in raw_chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue    
                cleaned_text = re.sub(r"^Subprompt\s*\d+:\s*", "", chunk, flags=re.IGNORECASE).strip()
                if cleaned_text:
                    current_subprompts.append(cleaned_text)
            if not current_subprompts:
                fallback = content_section.replace("Recaptioned subprompts:", "").strip()
                if fallback:
                    current_subprompts.append(fallback)
            
            batch_results.append(current_subprompts)
        return batch_results

    async def process_batch(
        self, tasks: List[RoundTask], states: List[RequestState]
    ) -> Optional[List[RoundTask]]:
        logger.info(f"[RPG] Processing batch: {len(tasks)} tasks")

        states_needing_init = [s for s in states if s.payload["recaption"] is None]

        if states_needing_init:
            original_queries = [s.original_query for s in states_needing_init]
            n_s = [s.config["n"] for s in states_needing_init]            

            recaption_list = await ModelRegistry.llm.async_generate_recaption(
                original_queries,
                n_s,
                sampling_params={"max_new_tokens": 2048},
            )
            for s, rw in zip(states_needing_init, recaption_list):
                s.payload["recaption"] = rw
                s.payload["subprompts"] = self.parse_llm_output([rw])[0]

        # --- 2. Planning ---
        system_prompt = ModelRegistry.llm.plan_prompt
        user_prompt_template = "Given Subprompt: {original_query}\n"
        decomposition_size = max([s.payload["decomposition_size"] for s in states])
        for _ in range(decomposition_size):
            rewrite_prompts = []
            for s in states:
                if s.payload["planning_history"]:
                    prompt = s.payload["planning_history"]
                else:
                    prompt = ModelRegistry.llm.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": user_prompt_template.format(original_query=s.payload["recaption"]),
                            },
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    s.payload["planning_history"] = prompt
                    
                rewrite_prompts.append(prompt)

            planning_list = await ModelRegistry.llm.async_generate(
                rewrite_prompts,
                sampling_params={"max_new_tokens": 2048, "stop": "====="},
            )

            for s, pl in zip(states, planning_list):
                s.payload["planning_history"] += f"{pl}====="

        # --- 3. Generate ---
        generate_inputs = []
        for s in states:
            for i in range(s.payload["decomposition_size"]):
                generate_inputs.append(s.payload["subprompts"][i])
        logger.info(f"[RPG] Generating images for {len(generate_inputs)} subprompts.")
        images = ModelRegistry.diffusion.generate(
            prompt=generate_inputs,
            height=512,
            width=512,
            num_inference_steps=50
        )


        for task, state in zip(tasks, states):
            state.current_round_idx += 1

            if task.is_last_round:
                state.round_answers.append({"answer": images})
                state.status = "finished"
                state.finished_at = time.perf_counter()
                GLOBAL_FINISHED_REQUEST_STATES[state.req_id] = state
                logger.info(
                    f"[Req {state.req_id}] ✅ Finished. Metrics: {state.metrics}. answer={state.payload['subprompts']}"
                )
            else:
                state.round_answers.append({"answer": "(Generated)"})
                state.priority -= 1
                await GLOBAL_REQUEST_STATES_QUEUE.put(state)
                logger.info(
                    f"[Request {state.req_id}] Round {task.round_idx} done."
                )
