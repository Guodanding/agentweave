import math
import time
import joblib
import logging
import numpy as np
from typing import List, Dict
from agentweave.utils.config import app_config, load_t_overhead
from agentweave.scheduler.state import RequestState

logger = logging.getLogger(__name__)

# ==========================================
# ==========================================
T_OVERHEAD = load_t_overhead("rag")


def load_profile(model_path):
    try:
        loaded_package = joblib.load(model_path)
    except FileNotFoundError:
        print("Model file not found. Run the training code first.")
        raise ValueError("not found profile model")
        return None
    poly_reg = loaded_package["poly_transformer"]
    model = loaded_package["linear_model"]
    return model, poly_reg


llm_profile = load_profile(app_config.args["rag"]["profile_llm_path"])
embd_profile = load_profile(app_config.args["rag"]["profile_embed_path"])
rerank_profile = load_profile(app_config.args["rag"]["profile_reranker_path"])


def predict_latency(profile, batch_size, input_len):
    new_data = np.array([[batch_size, input_len]])
    model, poly_reg = profile
    new_data_poly = poly_reg.transform(new_data)
    predicted_latency = model.predict(new_data_poly)
    return predicted_latency[0]


def decomposition_size_algorithm_rag(
    requests: List[RequestState],
) -> List[int]:
    # -----------------------------------------------------------
    # -----------------------------------------------------------
    req_data = []
    total_queue_tokens = 0
    log_length = {}

    for idx, req in enumerate(requests):
        num_text = len(req.payload["doc_token"])
        if num_text > 0:
            total_queue_tokens += sum(
                [len(token) for token in req.payload["doc_token"]]
            )
            req_data.append(
                {
                    "original_index": idx,
                    "req": req,
                    "num_text": num_text,
                    "token_length": [len(token) for token in req.payload["doc_token"]],
                    "config": req.config,
                }
            )
        log_length[req.req_id] = num_text

    if not req_data:
        return [0 for _ in requests]

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    lengths = [d["num_text"] for d in req_data]
    max_len = max(lengths)
    start = min(max_len, min(d["config"]["top_k"] for d in req_data))
    step = 1
    # step = max(1, int(max_len / 50))
    candidate_sizes = range(start, max_len + 1, step)

    best_final_score = -float("inf")
    best_decomposition_size = start  # Default safe value

    # -----------------------------------------------------------
    # -----------------------------------------------------------

    lambda_fairness = 0
    for decomposition_size in reversed(candidate_sizes):
        selected_tokens_list = []
        configs_list = []
        processed_requests_gain = []

        for i in range(len(req_data)):
            current_tokens = req_data[i]["token_length"][:decomposition_size]
            selected_tokens_list.append(current_tokens)

            req_total_tokens = sum(req_data[i]["token_length"])
            chunk_tokens = sum(current_tokens)
            processed_requests_gain.append(chunk_tokens / req_total_tokens)

            configs_list.append(req_data[i]["config"])

        throughput_gain = sum(processed_requests_gain)

        compute_latency, _, _, _ = calculate_simulated_metrics_rag(
            selected_tokens_list, configs_list, decomposition_size
        )

        total_latency = compute_latency + T_OVERHEAD

        safe_latency = max(total_latency, 0.0001)

        throughput_score = throughput_gain / safe_latency

        total_future_tax = 0.0
        for i in range(len(req_data)):
            num_chunks_needed = math.ceil(req_data[i]["num_text"] / decomposition_size)

            request_tax = num_chunks_needed * T_OVERHEAD
            total_future_tax += request_tax

        lambda_fairness = (
            len(requests) / compute_latency**2
            if lambda_fairness == 0
            else lambda_fairness
        )
        logger.info(f"Lambda Fairness: {lambda_fairness:.4f}")
        final_score = throughput_score - (lambda_fairness * total_future_tax)

        if final_score > best_final_score:
            best_final_score = final_score
            best_decomposition_size = decomposition_size

            # print(f"Update Best: Size={decomposition_size}, TP_Score={throughput_score:.2f}, Tax={total_future_tax:.4f}, Final={final_score:.2f}")
            # logger.info(f"Best: Decomposition Size={decomposition_size}, TP_Score={throughput_score:.2f}, Tax={total_future_tax:.4f}, Final={final_score:.2f}")
    logger.info(f"Best: Decomposition Size={best_decomposition_size}")
    # -----------------------------------------------------------
    # -----------------------------------------------------------
    return [min(best_decomposition_size, len(req.payload["doc_text"])) for req in requests]


def calculate_simulated_metrics_rag(
    selected_tokens_list: List[List[int]], configs_list: List[Dict], decomposition_size: int
) -> float:

    latency = 0.0

    # 1. Embedding
    batch_size = sum(len(x) for x in selected_tokens_list)
    all_tokens = [t for sublist in selected_tokens_list for t in sublist]
    input_len = np.mean(all_tokens) if all_tokens else 0
    embd_latency = predict_latency(embd_profile, batch_size, input_len)

    # 2. Rerank
    rerank_batch_size = sum([c["top_k"] for c in configs_list])
    rerank_input_len = input_len
    rerank_latency = predict_latency(
        rerank_profile, rerank_batch_size, rerank_input_len
    )

    # 3. LLM
    llm_batch_size = len(selected_tokens_list)
    llm_input_lens = [
        np.mean(tk) * c["rerank_top_k"]
        for tk, c in zip(selected_tokens_list, configs_list)
    ]
    llm_input_len_avg = np.mean(llm_input_lens) if llm_input_lens else 0
    llm_latency = predict_latency(llm_profile, llm_batch_size, llm_input_len_avg)

    latency = embd_latency + rerank_latency + llm_latency

    return latency, embd_latency, rerank_latency, llm_latency
