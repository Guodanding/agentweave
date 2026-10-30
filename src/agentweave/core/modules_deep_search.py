import math
import asyncio
import logging
import sglang as sgl
from typing import List, Union, Dict, Optional
from transformers import AutoTokenizer
from agentweave.utils.logger import force_setup_logging
from agentweave.utils.config import app_config


logger = logging.getLogger(__name__)


class LLM:

    def __init__(self, model_path: str, mem_fraction_static: float = 0.4):
        logger.info(f"Loading LLM (Async): {model_path}")
        self.engine = sgl.Engine(
            model_path=model_path,
            tp_size=2,
            is_embedding=False,
            mem_fraction_static=mem_fraction_static,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.synthesize_postfix = "\n---------------------\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    async def async_generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[Union[Dict, List[Dict]]] = None,
    ) -> Union[str, List[str]]:
        if not prompts:
            return [] if isinstance(prompts, list) else ""

        is_single_prompt = isinstance(prompts, str)
        prompts_list = [prompts] if is_single_prompt else prompts

        params_list = []
        if isinstance(sampling_params, dict) or sampling_params is None:
            base_params = sampling_params if sampling_params else {}
            params_list = [base_params for _ in prompts_list]
        elif isinstance(sampling_params, list):
            if len(sampling_params) != len(prompts_list):
                raise ValueError("sampling_params count must match prompts count.")
            params_list = sampling_params

        logger.info(f"Sending {len(prompts_list)} prompts to LLM async_generate.")
        try:
            outputs = await asyncio.wait_for(
                self.engine.async_generate(
                    prompt=prompts_list, sampling_params=params_list
                ),
                timeout=60.0,
            )
        except asyncio.TimeoutError:
            logger.error("LLM Inference Timed Out! Possible OOM or Deadlock.")
            exit()

        responses = [output["text"] for output in outputs]

        logger.info(f"LLM Generated {len(responses)} responses.")
        return responses[0] if is_single_prompt else responses

    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[Union[Dict, List[Dict]]] = None,
    ) -> Union[str, List[str]]:
        if not prompts:
            return [] if isinstance(prompts, list) else ""

        is_single_prompt = isinstance(prompts, str)
        prompts_list = [prompts] if is_single_prompt else prompts

        params_list = []
        if isinstance(sampling_params, dict) or sampling_params is None:
            base_params = sampling_params if sampling_params else {}
            params_list = [base_params for _ in prompts_list]
        elif isinstance(sampling_params, list):
            if len(sampling_params) != len(prompts_list):
                raise ValueError("sampling_params count must match prompts count.")
            params_list = sampling_params

        outputs = self.engine.generate(prompt=prompts_list, sampling_params=params_list)

        responses = [output["text"] for output in outputs]

        logger.info(f"LLM Generated {len(responses)} responses.")
        return responses[0] if is_single_prompt else responses

    async def async_generate_rewrite_query(
        self,
        original_queries: Union[str, List[str]],
        sampling_params: Optional[Union[Dict, List[Dict]]] = None,
    ):
        is_list = isinstance(original_queries, list)
        queries = original_queries if is_list else [original_queries]

        system_prompt = "You are a helpful assistant that rewrites a user's query..."
        user_prompt_template = "Rewrite the following query: {original_query}\n"

        rewrite_prompts = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt_template.format(original_query=q),
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for q in queries
        ]

        result = await self.async_generate(rewrite_prompts, sampling_params)
        return result

    def generate_rewrite_query(
        self,
        original_queries: Union[str, List[str]],
        sampling_params: Optional[Union[Dict, List[Dict]]] = None,
    ):
        is_list = isinstance(original_queries, list)
        queries = original_queries if is_list else [original_queries]

        system_prompt = "You are a helpful assistant that rewrites a user's query..."
        user_prompt_template = "Rewrite the following query: {original_query}\n"

        rewrite_prompts = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt_template.format(original_query=q),
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for q in queries
        ]

        result = self.generate(rewrite_prompts, sampling_params)
        return result

    def shutdown(self):
        self.engine.shutdown()


class ModelRegistry:

    llm: LLM
    small_llm: LLM

    @classmethod
    def load_all(cls):
        dr_config = app_config.args["deep_search"]
        cls.llm = LLM(
            dr_config["llm_path"],
            mem_fraction_static=dr_config["llm_mem_fraction_static"],
        )
        force_setup_logging()
        cls.small_llm = LLM(
            dr_config["small_llm_path"],
            mem_fraction_static=dr_config["small_llm_mem_fraction_static"],
        )
        force_setup_logging()
        logger.info("✅ All Models loaded in ModelRegistry.")

    @classmethod
    def shutdown_all(cls):
        if cls.llm:
            cls.llm.shutdown()
