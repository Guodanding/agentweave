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
T_OVERHEAD = load_t_overhead("kb_rag")

def load_profile(model_path):
    try:
        loaded_package = joblib.load(model_path)
    except FileNotFoundError:
        raise ValueError("not found profile model")
        return None
    poly_reg = loaded_package["poly_transformer"]
    model = loaded_package["linear_model"]
    return model, poly_reg


llm_profile = load_profile(app_config.args["kb_rag"]["profile_llm_path"])
embd_profile = load_profile(app_config.args["kb_rag"]["profile_embed_path"])
rerank_profile = load_profile(app_config.args["kb_rag"]["profile_reranker_path"])


def predict_latency(profile, batch_size, input_len):
    new_data = np.array([[batch_size, input_len]])
    model, poly_reg = profile
    new_data_poly = poly_reg.transform(new_data)
    predicted_latency = model.predict(new_data_poly)
    return predicted_latency[0]


def decomposition_size_algorithm_kb_rag(
    requests: List[RequestState],
) -> List[int]:
    # -----------------------------------------------------------
    # -----------------------------------------------------------
    req_data = []
    log_length = {}

    for idx, req in enumerate(requests):
        remaining_k = req.payload["remaining_k"]
        if remaining_k > 0:
            req_data.append(
                {
                    "original_index": idx,
                    "req": req,
                    "remaining_k": remaining_k,
                    "config": req.config,
                }
            )
        log_length[req.req_id] = remaining_k

    if not req_data:
        return [0 for _ in requests]

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    lengths = [d["remaining_k"] for d in req_data]
    max_len = max(lengths)
    start = min(lengths)
    step = 1
    # step = max(1, int(max_len / 50))
    candidate_sizes = range(start, max_len + 1, step)

    best_final_score = -float("inf")
    best_decomposition_size = start  # Default safe value

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    # print("Total Chunk: ", max_len)

    lambda_fairness = 0
    for decomposition_size in reversed(candidate_sizes):
        selected_ks_list = []
        configs_list = []
        processed_requests_gain = []

        for i in range(len(req_data)):
            current_k = min(req_data[i]["remaining_k"], decomposition_size)
            selected_ks_list.append(current_k)

            processed_requests_gain.append(current_k / req_data[i]["remaining_k"])

            configs_list.append(req_data[i]["config"])

        throughput_gain = sum(processed_requests_gain)

        compute_latency, _, _, _ = calculate_simulated_metrics_kb_rag(
            selected_ks_list, configs_list
        )

        total_latency = compute_latency + T_OVERHEAD

        safe_latency = max(total_latency, 0.0001)

        throughput_score = throughput_gain / safe_latency

        total_future_tax = 0.0
        for i in range(len(req_data)):
            num_chunks_needed = math.ceil(req_data[i]["remaining_k"] / decomposition_size)

            request_tax = num_chunks_needed * T_OVERHEAD
            total_future_tax += request_tax

        lambda_fairness = (
            len(requests) / compute_latency**2
            if lambda_fairness == 0
            else lambda_fairness
        )
        # logger.info(f"Lambda Fairness: {lambda_fairness:.4f}")
        final_score = throughput_score - (lambda_fairness * total_future_tax)

        if final_score > best_final_score:
            best_final_score = final_score
            best_decomposition_size = decomposition_size

            # print(f"Update Best: Size={decomposition_size}, TP_Score={throughput_score:.2f}, Tax={total_future_tax:.4f}, Final={final_score:.2f}")
            # logger.info(f"Best: Decomposition Size={decomposition_size}, TP_Score={throughput_score:.2f}, Tax={total_future_tax:.4f}, Final={final_score:.2f}")
    logger.info(f"Best: Decomposition Size={best_decomposition_size}")
    # -----------------------------------------------------------
    # -----------------------------------------------------------
    return [min(best_decomposition_size, max(0, req.payload["remaining_k"])) for req in requests]


def calculate_simulated_metrics_kb_rag(
    selected_ks_list: List[int], configs_list: List[Dict]
) -> float:

    latency = 0.0
    rerank_batch_size = sum(selected_ks_list)
    rerank_input_len = 1024
    rerank_latency = predict_latency(
        rerank_profile, rerank_batch_size, rerank_input_len
    )

    llm_batch_size = len(selected_ks_list)
    llm_input_lens = [
        1024 * min(c["rerank_top_k"], ks)
        for ks, c in zip(selected_ks_list, configs_list)
    ]
    llm_input_len_avg = np.mean(llm_input_lens) if llm_input_lens else 0
    llm_latency = predict_latency(llm_profile, llm_batch_size, llm_input_len_avg)

    latency = rerank_latency + llm_latency

    return latency, None, rerank_latency, llm_latency
