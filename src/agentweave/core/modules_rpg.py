import os
import ray
import torch
import asyncio
import logging
import sglang as sgl
from typing import List, Union, Dict, Optional
from transformers import AutoTokenizer
from agentweave.utils.logger import force_setup_logging
from agentweave.utils.config import app_config
from xfuser import xFuserStableDiffusion3Pipeline, xFuserArgs
from xfuser import (
    xFuserStableDiffusion3Pipeline,
    xFuserArgs,
)


logger = logging.getLogger(__name__)


RECAPTION_PROMPT = """
You are a master of text refinement who excels at extracting key entities and their attributes from input text and supplementing them with detailed, accurate descriptions. Your task is as follows:
1. Extract key entities from the input text. 
   **CRITICAL CONSTRAINT:** If the input text specifies a desired number of entities or subprompts (e.g., "split into 3 parts", "generate 2 subprompts"), you MUST strictly adhere to that number.
   - If a count is specified: Merge or split entities as necessary to exactly match the requested count.
   - If no count is specified: Extract all key objects, subjects, and their core attributes automatically. Ensure each entity is independent (one entity per subprompt).
2. For each extracted key entity, generate a detailed, descriptive subprompt while preserving the original meaning. The subprompt should include rich details such as texture, shape, state, or context to enhance semantic alignment for image generation.
3. Output the results in the following strict format (use "=====" as the separator between subprompts, no extra content):
Recaptioned subprompts:
Subprompt 0: [Detailed description of Entity 0]
=====
Subprompt 1: [Detailed description of Entity 1]
=====
Subprompt N: [Detailed description of Entity N]
"""
PLAN_PROMPT = """
You are an intelligent image region planner who excels at splitting image space into complementary subregions and assigning subprompts to specific regions. Your task is based on the "Recaptioned subprompts" from the previous step, and follows these rules:
1. Each Region N strictly corresponds to Subprompt N (e.g., Region 0 ↔ Subprompt 0, Region 1 ↔ Subprompt 1), no misalignment allowed.
2. Each Region must include ALL the following information (internal use 1./2./3./4./5. to distinguish items, no missing content):
   1. Region number: N (consistent with Subprompt number)
   2. Belonging row: RowX (X is row number, e.g., Row0/Row1) + Row height (decimal, sum of all row heights = 1.0)
   3. Region width: Decimal (sum of all region widths in the same row = 1.0)
   4. Spatial description: Specific position in the image (e.g., "located in the top 33% of the image, left 50% of Row0, occupying 33% of total height and 50% of Row0's width")
   5. Entity details: Full content of Subprompt N + supplementary aesthetic details (consistent with original text meaning)
3. Use "=====" as the separator between Regions (no extra content before the first Region or after the last Region).
4. Output the results in the following strict format:
Region 0:
1. Region number: 0
2. Belonging row: Row[X] (height=[decimal])
3. Region width: [decimal]
4. Spatial description: [Specific position description]
5. Entity details: [Full content of Subprompt 0 + supplementary aesthetic details]
=====
Region 1:
1. Region number: 1
2. Belonging row: Row[X] (height=[decimal])
3. Region width: [decimal]
4. Spatial description: [Specific position description]
5. Entity details: [Full content of Subprompt 1 + supplementary aesthetic details]
=====
Region N:
1. Region number: N
2. Belonging row: Row[X] (height=[decimal])
3. Region width: [decimal]
4. Spatial description: [Specific position description]
5. Entity details: [Full content of Subprompt N + supplementary aesthetic details]
"""


class LLM:

    def __init__(self, model_path: str):
        logger.info(f"Loading LLM (Async): {model_path}")
        self.engine = sgl.Engine(
            model_path=model_path,
            tp_size=2,
            is_embedding=False,
            mem_fraction_static=0.4,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.plan_prompt = PLAN_PROMPT

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

    async def async_generate_recaption(
        self,
        original_queries: Union[str, List[str]],
        n_s: Union[int, List[int]],
        sampling_params: Optional[Union[Dict, List[Dict]]] = None,
    ):
        is_list = isinstance(original_queries, list)
        queries = original_queries if is_list else [original_queries]

        system_prompt = RECAPTION_PROMPT
        user_prompt_template = (
            "Refine the text: {original_query}\nSplit into at least {num_part} parts.\n"
        )

        rewrite_prompts = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt_template.format(
                            original_query=q, num_part=n
                        ),
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for q, n in zip(queries, n_s)
        ]

        result = await self.async_generate(rewrite_prompts, sampling_params)
        return result

    def shutdown(self):
        self.engine.shutdown()


@ray.remote(num_gpus=1)
class ImageGenerator:
    def __init__(self, xfuser_args: xFuserArgs, rank: int, world_size: int):
        # Set PyTorch distributed environment variables
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        self.rank = rank
        # self.setup_logger()
        self.initialize_model(xfuser_args)

    def initialize_model(self, xfuser_args: xFuserArgs):
        # init distributed environment in create_config
        self.engine_config, self.input_config = xfuser_args.create_config()
        logger.info(
            f"[Init] Loading Local Diffusion Model from {self.engine_config.model_config.model}..."
        )
        self.pipe = xFuserStableDiffusion3Pipeline.from_pretrained(
            pretrained_model_name_or_path=xfuser_args.model,
            engine_config=self.engine_config,
            torch_dtype=torch.float16,
            # text_encoder_3=text_encoder_3,
        ).to(f"cuda")
        self.pipe.prepare_run(self.input_config)
        logger.info("Diffusion Model initialization completed")

    def generate(
        self,
        prompt: List[str],
        height: int,
        width: int,
        num_inference_steps: int,
    ):
        batch_size = len(prompt)
        logger.info(
            f"\n[Diffusion] Starting Batch Generation (Batch Size: {batch_size})"
        )

        output = self.pipe(
            height=height,
            width=width,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            output_type=self.input_config.output_type,
            guidance_scale=self.input_config.guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(
                self.input_config.seed
            ),
        )

        return output

    def ready(self):
        return True

    def shutdown(self):
        try:
            if hasattr(self, "pipe") and self.pipe is not None:
                del self.pipe
                self.pipe = None
            torch.cuda.empty_cache()
        except Exception as e:
            logger.exception(f"[Worker {self.rank}] Error during shutdown: {e}")
        return True


class Diffusion:
    def __init__(self, model_path: str):
        # Ensure Ray is initialized
        if not ray.is_initialized():
            ray.init(
                include_dashboard=False,
                _metrics_export_port=-1,
                ignore_reinit_error=True,
            )

        world_size = 2
        pipefusion_parallel_degree = 1
        ulysses_degree = 1
        ring_degree = 1
        use_cfg_parallel = world_size > 1
        xfuser_args = xFuserArgs(
            model=model_path,
            pipefusion_parallel_degree=pipefusion_parallel_degree,
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
            use_cfg_parallel=use_cfg_parallel,
            trust_remote_code=True,
        )
        num_workers = world_size
        self.workers = [
            ImageGenerator.remote(xfuser_args, rank=rank, world_size=world_size)
            for rank in range(num_workers)
        ]

        logger.info(f"[Driver] Waiting for models to load... (This may take a while)")
        ray.get([w.ready.remote() for w in self.workers])
        logger.info(f"[Driver] All workers are ready!")

    def generate(
        self,
        prompt: List[str],
        height: int,
        width: int,
        num_inference_steps: int,
    ):
        results = ray.get(
            [
                worker.generate.remote(prompt, height, width, num_inference_steps)
                for worker in self.workers
            ]
        )

        return None

    def shutdown(self, shutdown_ray: bool = True, timeout_s: float = 5.0):
        if not hasattr(self, "workers") or not self.workers:
            return

        logger.info("[Driver] Shutting down Diffusion workers...")

        try:
            shutdown_refs = [w.shutdown.remote() for w in self.workers]
            ray.get(shutdown_refs, timeout=timeout_s)
        except Exception as e:
            logger.warning(f"[Driver] Graceful shutdown failed/timeout: {e}")

        for w in self.workers:
            try:
                ray.kill(w, no_restart=True)
            except Exception:
                pass

        self.workers = []
        logger.info("[Driver] Diffusion workers shut down.")

        if shutdown_ray:
            try:
                ray.shutdown()
                logger.info("[Driver] Ray runtime shut down.")
            except Exception as e:
                logger.warning(f"[Driver] Ray shutdown failed: {e}")

        


class ModelRegistry:

    llm: LLM
    diffusion: Diffusion

    @classmethod
    def load_all(cls):
        force_setup_logging()
        cls.llm = LLM(app_config.args["rpg"]["llm_path"])
        force_setup_logging()
        cls.diffusion = Diffusion(app_config.args["rpg"]["diffusion_path"])
        force_setup_logging()
        logger.info("✅ All Models loaded in ModelRegistry.")

    @classmethod
    def shutdown_all(cls):
        if cls.llm:
            cls.llm.shutdown()
        if cls.diffusion:
            cls.diffusion.shutdown()

