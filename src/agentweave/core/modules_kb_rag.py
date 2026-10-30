import math
import asyncio
import logging
import sglang as sgl
from typing import List, Union, Dict, Optional, Any
from transformers import AutoTokenizer
from agentweave.utils.logger import force_setup_logging
from agentweave.utils.config import app_config
import pickle
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class LLM:

    def __init__(self, model_path: str):
        logger.info(f"Loading LLM (Async): {model_path}")
        self.engine = sgl.Engine(
            model_path=model_path,
            tp_size=4,
            is_embedding=False,
            mem_fraction_static=0.4,
            context_length=128*1024,
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


class Embedding:

    def __init__(self, model_path: str):
        logger.info(f"Loading Embedding Model (Async): {model_path}")
        self.engine = sgl.Engine(
            model_path=model_path,
            tp_size=4,
            is_embedding=True,
            mem_fraction_static=0.3,
        )

    async def async_encode(
        self,
        prompts: Union[str, List[str]],
    ) -> Union[List[float], List[List[float]]]:
        if not prompts:
            return []
        is_single_prompt = isinstance(prompts, str)
        prompts = [prompts] if is_single_prompt else prompts

        outputs = await self.engine.async_encode(prompts)

        embeddings = [output["embedding"] for output in outputs]
        return embeddings[0] if is_single_prompt else embeddings

    def encode(
        self,
        prompts: Union[str, List[str]],
    ) -> Union[List[float], List[List[float]]]:
        if not prompts:
            return []
        is_single_prompt = isinstance(prompts, str)
        prompts = [prompts] if is_single_prompt else prompts

        outputs = self.engine.encode(prompts)

        embeddings = [output["embedding"] for output in outputs]
        return embeddings[0] if is_single_prompt else embeddings

    def shutdown(self):
        self.engine.shutdown()


class QwenReranker:

    def __init__(self, model_path: str):
        logger.info(f"Loading Reranker (Async): {model_path}")
        self.engine = sgl.Engine(
            model_path=model_path,
            tp_size=4,
            mem_fraction_static=0.6,
            disable_radix_cache=True,
            context_length=128*1024,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.yes_token_id = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.no_token_id = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        self.yes_token_id_str = str(self.yes_token_id)
        self.no_token_id_str = str(self.no_token_id)

    def _create_template_prompts(
        self, prompts: List[List[str]], instruction: str
    ) -> List[str]:
        prompts_as_dicts = []
        for query, doc in prompts:
            formatted_prompt = [
                {
                    "role": "system",
                    "content": 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".',
                },
                {
                    "role": "user",
                    "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}",
                },
            ]
            prompts_as_dicts.append(formatted_prompt)

        return self.tokenizer.apply_chat_template(
            prompts_as_dicts,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    async def async_rerank(
        self, prompts: List[List[str]], instruction: str
    ) -> List[float]:
        if not prompts:
            return []

        template_prompts = self._create_template_prompts(prompts, instruction)
        token_ids_to_request = [[self.yes_token_id, self.no_token_id] for _ in prompts]

        sampling_params = {
            "max_new_tokens": 1,
        }

        outputs = await self.engine.async_generate(
            prompt=template_prompts,
            sampling_params=sampling_params,
            return_logprob=True,
            token_ids_logprob=token_ids_to_request,
        )

        scores = []
        for output in outputs:
            try:
                logprobs_tuples = output["meta_info"]["output_token_ids_logprobs"][0]
                logprob_dict = {
                    str(token_id): logprob for logprob, token_id, _ in logprobs_tuples
                }
                yes_logit = logprob_dict.get(self.yes_token_id_str, -100.0)
                no_logit = logprob_dict.get(self.no_token_id_str, -100.0)
                yes_prob = math.exp(yes_logit)
                no_prob = math.exp(no_logit)
                total_prob = yes_prob + no_prob
                score = yes_prob / total_prob if total_prob > 0 else 0.0
                scores.append(score)
            except (KeyError, IndexError):
                logger.error(
                    f"Could not find 'output_token_ids_logprobs' in the output. Output: {output}"
                )
                scores.append(0.0)
        return scores

    def rerank(self, prompts: List[List[str]], instruction: str) -> List[float]:
        if not prompts:
            return []

        template_prompts = self._create_template_prompts(prompts, instruction)
        token_ids_to_request = [[self.yes_token_id, self.no_token_id] for _ in prompts]

        sampling_params = {
            "max_new_tokens": 1,
        }

        outputs = self.engine.generate(
            prompt=template_prompts,
            sampling_params=sampling_params,
            return_logprob=True,
            token_ids_logprob=token_ids_to_request,
        )

        scores = []
        for output in outputs:
            try:
                logprobs_tuples = output["meta_info"]["output_token_ids_logprobs"][0]
                logprob_dict = {
                    str(token_id): logprob for logprob, token_id, _ in logprobs_tuples
                }
                yes_logit = logprob_dict.get(self.yes_token_id_str, -100.0)
                no_logit = logprob_dict.get(self.no_token_id_str, -100.0)
                yes_prob = math.exp(yes_logit)
                no_prob = math.exp(no_logit)
                total_prob = yes_prob + no_prob
                score = yes_prob / total_prob if total_prob > 0 else 0.0
                scores.append(score)
            except (KeyError, IndexError):
                logger.error(
                    f"Could not find 'output_token_ids_logprobs' in the output. Output: {output}"
                )
                scores.append(0.0)
        return scores

    def shutdown(self):
        self.engine.shutdown()


class ModelRegistry:

    llm: LLM
    embed: Embedding
    reranker: QwenReranker
    global_faiss_index: faiss.Index
    global_nodes_mapping: Dict[int, Any]

    @classmethod
    def load_all(cls):
        force_setup_logging()
        cls.llm = LLM(app_config.args["kb_rag"]["llm_path"])
        force_setup_logging()
        cls.embed = Embedding(app_config.args["kb_rag"]["embed_path"])
        force_setup_logging()
        cls.reranker = QwenReranker(app_config.args["kb_rag"]["reranker_path"])
        force_setup_logging()
        cls.global_faiss_index = faiss.read_index(app_config.args["kb_rag"]["index_path"])
        with open(app_config.args["kb_rag"]["metadata_path"], "rb") as f:
            cls.global_nodes_mapping = pickle.load(f)
        logger.info("Global FAISS Index.")
        logger.info("✅ All Models loaded in ModelRegistry.")


    @classmethod
    def shutdown_all(cls):
        if cls.llm:
            cls.llm.shutdown()
        if cls.embed:
            cls.embed.shutdown()
        if cls.reranker:
            cls.reranker.shutdown()
