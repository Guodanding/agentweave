import os
import joblib
import pickle
import math
from pathlib import Path
from typing import Dict
from dotenv import find_dotenv, load_dotenv
from types import SimpleNamespace
import argparse

DOTENV_PATH = find_dotenv()
load_dotenv(DOTENV_PATH)


def load_t_overhead(agent_type: str) -> float:
    file_path = os.getenv("T_OVERHEAD_FILE")
    if not file_path or not file_path.strip():
        raise ValueError("T_OVERHEAD_FILE must specify an overhead profile file.")
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        base_dir = Path(DOTENV_PATH).resolve().parent if DOTENV_PATH else Path.cwd()
        path = base_dir / path
    try:
        values = joblib.load(path)
    except (OSError, ValueError, EOFError, pickle.UnpicklingError, IndexError, KeyError) as exc:
        raise ValueError(f"Cannot read T_OVERHEAD_FILE {path}: {exc}") from exc
    if not isinstance(values, dict) or agent_type not in values:
        raise ValueError(f"T_OVERHEAD_FILE {path} must contain a value for {agent_type}.")
    value = values[agent_type]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"T_OVERHEAD for {agent_type} must be a number in seconds.")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"T_OVERHEAD for {agent_type} must be finite and non-negative.")
    return float(value)


class AppConfig:
    _instance = None
    _parsed_args = None
    _args = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def args(self) -> Dict:
        if self._args is None:
            args = self._get_parsed_args()
            self._args = self._build_agent_configs(args)
        return self._args

    def _get_parsed_args(self):
        if self._parsed_args is None:
            self._parsed_args = parse_args()
        return self._parsed_args

    def _build_agent_configs(self, args):
        return_args = {}
        return_args["agent_type"] = getattr(args, "agent_type", None) or os.getenv(
            "AGENT_TYPE"
        )
        return_args["batch_size"] = int(
            getattr(args, "batch_size", None) or os.getenv("BATCH_SIZE", 10)
        )

        return_args["rag"] = {
            "llm_path": getattr(args, "rag_llm_path", None)
            or os.getenv("RAG_LLM_PATH"),
            "embed_path": getattr(args, "rag_embed_path", None)
            or os.getenv("RAG_EMBED_PATH"),
            "reranker_path": getattr(args, "rag_reranker_path", None)
            or os.getenv("RAG_RERANKER_PATH"),
            # "decomposition_size": int(
            #     getattr(args, "rag_decomposition_size", None)
            #     or os.getenv("RAG_DECOMPOSITION_SIZE", 40960)
            # ),
            "profile_llm_path": getattr(args, "rag_profile_llm_path", None)
            or os.getenv("RAG_PROFILE_LLM_PATH"),
            "profile_embed_path": getattr(args, "rag_profile_embed_path", None)
            or os.getenv("RAG_PROFILE_EMBED_PATH"),
            "profile_reranker_path": getattr(args, "rag_profile_reranker_path", None)
            or os.getenv("RAG_PROFILE_RERANKER_PATH"),
        }

        return_args["kb_rag"] = {
            "index_path": os.getenv("KB_INDEX_PATH"),
            "metadata_path": os.getenv("KB_METADATA_PATH"),
            "llm_path": getattr(args, "kb_llm_path", None)
            or os.getenv("KB_LLM_PATH"),
            "embed_path": getattr(args, "kb_embed_path", None)
            or os.getenv("KB_EMBED_PATH"),
            "reranker_path": getattr(args, "kb_reranker_path", None)
            or os.getenv("KB_RERANKER_PATH"),
            # "decomposition_size": int(
            #     getattr(args, "kb_decomposition_size", None)
            #     or os.getenv("KB_DECOMPOSITION_SIZE", 40960)
            # ),
            "profile_llm_path": getattr(args, "kb_profile_llm_path", None)
            or os.getenv("KB_PROFILE_LLM_PATH"),
            "profile_embed_path": getattr(args, "kb_profile_embed_path", None)
            or os.getenv("KB_PROFILE_EMBED_PATH"),
            "profile_reranker_path": getattr(args, "kb_profile_reranker_path", None)
            or os.getenv("KB_PROFILE_RERANKER_PATH"),
        }

        return_args["deep_search"] = {
            "llm_path": getattr(args, "dr_llm_path", None) or os.getenv("DR_LLM_PATH"),
            "small_llm_path": getattr(args, "dr_small_llm_path", None)
            or os.getenv("DR_SMALL_LLM_PATH"),
            "llm_mem_fraction_static": float(
                getattr(args, "dr_llm_mem_fraction_static", None)
                or os.getenv("DR_LLM_MEM_FRACTION_STATIC", 0.5)
            ),
            "small_llm_mem_fraction_static": float(
                getattr(args, "dr_small_llm_mem_fraction_static", None)
                or os.getenv("DR_SMALL_LLM_MEM_FRACTION_STATIC", 0.6)
            ),
            "profile_llm_path": getattr(args, "dr_profile_llm_path", None)
            or os.getenv("DR_PROFILE_LLM_PATH"),
            "profile_small_llm_path": getattr(args, "dr_profile_small_llm_path", None)
            or os.getenv("DR_PROFILE_SMALL_LLM_PATH"),
            "profile_small_llm_decode_path": getattr(args, "dr_profile_small_llm_decode_path", None)
            or os.getenv("DR_PROFILE_SMALL_LLM_DECODE_PATH"),
            "subagent_concurrency_limit": int(
                getattr(args, "dr_subagent_concurrency_limit", None)
                or os.getenv("DR_SUBAGENT_CONCURRENCY_LIMIT", 4)
            ),
            "search_top_k": int(
                getattr(args, "dr_search_top_k", None)
                or os.getenv("DR_SEARCH_TOP_K", 10)
            ),
        }

        return_args["rpg"] = {
            "llm_path": getattr(args, "rpg_llm_path", None)
            or os.getenv("RPG_LLM_PATH"),
            "diffusion_path": getattr(args, "rpg_diffusion_path", None)
            or os.getenv("RPG_DIFFUSION_PATH"),
            "profile_llm_path": getattr(args, "rpg_profile_llm_path", None)
            or os.getenv("RPG_PROFILE_LLM_PATH"),
            "profile_diffusion_path": getattr(args, "rpg_profile_diffusion_path", None)
            or os.getenv("RPG_PROFILE_DIFFUSION_PATH"),
        }


        return return_args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with configurable agent (RAG, Deep Search, etc.)."
    )

    # parser.add_argument("--rag_llm_path", type=str, help="LLM path for RAG agent")
    # parser.add_argument(
    #     "--rag_embed_path", type=str, help="Embedding model path for RAG"
    # )
    # parser.add_argument(
    #     "--rag_reranker_path", type=str, help="Reranker model path for RAG"
    # )
    # parser.add_argument("--rag_decomposition_size", type=int, help="Decomposition size for RAG")
    #
    # parser.add_argument(
    #     "--dr_llm_path", type=str, help="LLM path for Deep Search agent"
    # )
    # parser.add_argument(
    #     "--dr_embed_path", type=str, help="Embedding model path for Deep Search"
    # )
    # parser.add_argument(
    #     "--dr_decomposition_size", type=int, help="Decomposition size for Deep Search"
    # )

    parser.add_argument(
        "--agent_type",
        type=str,
        help="Agent app to serve",
        choices=["rag", "kb_rag", "deep_search", "rpg"],
    )

    return parser.parse_args()


app_config = AppConfig()
