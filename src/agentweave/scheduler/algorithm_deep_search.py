
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
T_OVERHEAD = load_t_overhead("deep_search")


def load_profile(model_path):
    try:
        loaded_package = joblib.load(model_path)
    except FileNotFoundError:
        raise ValueError("not found profile model")
    poly_reg = loaded_package["poly_transformer"]
    model = loaded_package["linear_model"]
    return model, poly_reg


llm_profile = load_profile(app_config.args["deep_search"]["profile_llm_path"])
small_llm_profile = load_profile(
    app_config.args["deep_search"]["profile_small_llm_path"]
)
small_llm_decode_profile = load_profile(
    app_config.args["deep_search"]["profile_small_llm_decode_path"]
)


def predict_latency(profile, batch_size, input_len):
    if profile is None:
        return 0.1
    new_data = np.array([[batch_size, input_len]])
    model, poly_reg = profile
    new_data_poly = poly_reg.transform(new_data)
    predicted_latency = model.predict(new_data_poly)
    return predicted_latency[0]


def decomposition_size_algorithm_deep_search(requests: List[RequestState]) -> List[int]:
    # -----------------------------------------------------------
    # -----------------------------------------------------------
    if not requests:
        return []

    req_data = []
    log_length = {}

    for idx, req in enumerate(requests):
        num_subagents = req.config.get("num_subagents", 3)
        end_idx = req.payload.get("end_idx", 0)
        remaining_rounds = max(0, num_subagents - end_idx)

        if remaining_rounds > 0:
            req_data.append(
                {
                    "original_index": idx,
                    "req": req,
                    "remaining_rounds": remaining_rounds,
                    "config": req.config,
                }
            )
        log_length[req.req_id] = remaining_rounds

    if not req_data:
        return [3] * len(requests)

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    lengths = [d["remaining_rounds"] for d in req_data]
    max_len = 10
    start = 4
    step = 1

    candidate_sizes = range(start, max_len + 1, step)

    best_final_score = -float("inf")
    best_decomposition_size = start

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    lambda_fairness = 0
    for decomposition_size in reversed(candidate_sizes):
        processed_requests_gain = []
        actual_rounds_per_req = []
        configs_list = []

        for i in range(len(req_data)):
            current_chunk_rounds = min(decomposition_size, req_data[i]["remaining_rounds"])
            actual_rounds_per_req.append(current_chunk_rounds)

            req_total_rounds = req_data[i]["remaining_rounds"]
            processed_requests_gain.append(current_chunk_rounds / req_total_rounds)

            configs_list.append(req_data[i]["config"])

        throughput_gain = sum(processed_requests_gain)

        total_subagents_this_round = sum(actual_rounds_per_req)

        num_requests = len(req_data)
        (
            compute_latency,
            _,
            _,
        ) = calculate_simulated_metrics_deep_search(
            actual_rounds_per_req, configs_list
        )

        total_latency = compute_latency + T_OVERHEAD

        safe_latency = max(total_latency, 0.0001)

        throughput_score = throughput_gain / safe_latency

        total_future_tax = 0.0
        for i in range(len(req_data)):
            num_chunks_needed = math.ceil(req_data[i]["remaining_rounds"] / decomposition_size)

            request_tax = num_chunks_needed * T_OVERHEAD
            total_future_tax += request_tax

        lambda_fairness = (
            len(requests) / (compute_latency**2)
            if lambda_fairness == 0
            else lambda_fairness
        )
        final_score = throughput_score - (lambda_fairness * total_future_tax)

        if final_score > best_final_score:
            best_final_score = final_score
            best_decomposition_size = decomposition_size

    logger.info(
        f"[FATS-DR] Optimal Decomposition Size={best_decomposition_size}, Score={best_final_score:.4f}"
    )

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    return [best_decomposition_size for _ in range(len(requests))]


def calculate_simulated_metrics_deep_search(
    actual_rounds_per_req: List[int], configs_list: List[Dict]
) -> tuple:
    # ========================================
    # ========================================
    SEARCH_RESULT_TOKENS = 2048 * 8
    EVALUATION_TOKENS = 128
    SUBAGENT_INPUT_LEN = SEARCH_RESULT_TOKENS

    num_requests = len(actual_rounds_per_req)
    total_subagents = sum(actual_rounds_per_req)

    # ========================================
    # ========================================
    subagent_prefill_latency = predict_latency(
        small_llm_profile, total_subagents, SUBAGENT_INPUT_LEN
    )
    subagent_decode_latency = predict_latency(
        small_llm_decode_profile, total_subagents, EVALUATION_TOKENS
    )
    subagent_llm_latency = subagent_prefill_latency + subagent_decode_latency

    # ========================================
    # ========================================
    lead_prefill_latencies = []
    for rounds in actual_rounds_per_req:
        prefill_len = rounds * (SEARCH_RESULT_TOKENS + EVALUATION_TOKENS)
        lead_prefill_latencies.append(
            predict_latency(llm_profile, 1, prefill_len)
        )
    lead_llm_latency = sum(lead_prefill_latencies) / max(num_requests, 1)

    total_latency = lead_llm_latency + subagent_llm_latency
    return total_latency, subagent_llm_latency, lead_llm_latency
