import math
import joblib
import logging
import numpy as np
from typing import List, Dict
from agentweave.utils.config import app_config, load_t_overhead
from agentweave.scheduler.state import RequestState

logger = logging.getLogger(__name__)

# ==========================================
# ==========================================
T_OVERHEAD = load_t_overhead("rpg")


def load_profile(model_path):
    try:
        loaded_package = joblib.load(model_path)
    except FileNotFoundError:
        raise ValueError("not found profile model")
        return None
    poly_reg = loaded_package["poly_transformer"]
    model = loaded_package["linear_model"]
    return model, poly_reg


llm_profile = load_profile(app_config.args["rpg"]["profile_llm_path"])
diffusion_profile = load_profile(app_config.args["rpg"]["profile_diffusion_path"])


def predict_latency(profile, batch_size):
    new_data = np.array([[batch_size]])
    model, poly_reg = profile
    new_data_poly = poly_reg.transform(new_data)
    predicted_latency = model.predict(new_data_poly)
    return predicted_latency[0]


def decomposition_size_algorithm_rpg(
    requests: List[RequestState],
) -> List[int]:
    # -----------------------------------------------------------
    # -----------------------------------------------------------
    req_data = []
    log_length = {}

    for idx, req in enumerate(requests):
        remaining_n = req.payload["remaining_n"]
        if remaining_n > 0:
            req_data.append(
                {
                    "original_index": idx,
                    "req": req,
                    "remaining_n": remaining_n,
                    "config": req.config,
                }
            )
        log_length[req.req_id] = remaining_n

    if not req_data:
        return [], 0

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    lengths = [d["remaining_n"] for d in req_data]
    max_len = max(lengths)
    start = min(lengths)
    step = 1
    # step = max(1, int(max_len / 50))
    candidate_sizes = range(start, max_len + 1, step)

    best_final_score = -float("inf")
    best_decomposition_size = start  # Default safe value

    # -----------------------------------------------------------
    # -----------------------------------------------------------

    lambda_fairness = 0
    for decomposition_size in reversed(candidate_sizes):
        selected_ns_list = []
        processed_requests_gain = []

        for i in range(len(req_data)):
            current_n = min(req_data[i]["remaining_n"], decomposition_size)
            selected_ns_list.append(current_n)

            processed_requests_gain.append(current_n / req_data[i]["remaining_n"])

        throughput_gain = sum(processed_requests_gain)

        compute_latency, _, _ = calculate_simulated_metrics_rpg(selected_ns_list)

        total_latency = compute_latency + T_OVERHEAD

        safe_latency = max(total_latency, 0.0001)

        throughput_score = throughput_gain / safe_latency

        total_future_tax = 0.0
        for i in range(len(req_data)):
            num_chunks_needed = math.ceil(req_data[i]["remaining_n"] / decomposition_size)
            request_tax = num_chunks_needed * T_OVERHEAD
            total_future_tax += request_tax

        lambda_fairness = (
            len(requests) / compute_latency**2
            if lambda_fairness == 0
            else lambda_fairness
        )
        final_score = throughput_score - (lambda_fairness * total_future_tax)
        logger.info(
            f"Lambda Fairness: {lambda_fairness:.4f}, Decomposition Size: {decomposition_size}, Throughput Score: {throughput_score:.4f}, Latency: {total_latency:.4f}, Ns: {selected_ns_list}"
        )

        if final_score > best_final_score:
            best_final_score = final_score
            best_decomposition_size = decomposition_size

            # print(f"Update Best: Size={decomposition_size}, TP_Score={throughput_score:.2f}, Tax={total_future_tax:.4f}, Final={final_score:.2f}")
            # logger.info(f"Best: Decomposition Size={decomposition_size}, TP_Score={throughput_score:.2f}, Tax={total_future_tax:.4f}, Final={final_score:.2f}")
    logger.info(f"Best: Decomposition Size={best_decomposition_size}")
    # -----------------------------------------------------------
    # -----------------------------------------------------------
    assert len(requests) == len(req_data)
    return [
        min(best_decomposition_size, req_data[i]["remaining_n"]) for i in range(len(requests))
    ]


def calculate_simulated_metrics_rpg(selected_ns_list: List[int]) -> float:

    latency = 0.0
    batch_size = sum(selected_ns_list)
    # 1. LLM
    llm_latency = predict_latency(llm_profile, batch_size) * 128
    # 2. Diffusion
    diffusion_latency = predict_latency(diffusion_profile, batch_size)

    latency = llm_latency + diffusion_latency

    return latency, llm_latency, diffusion_latency

