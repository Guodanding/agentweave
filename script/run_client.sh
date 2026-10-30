#!/usr/bin/env bash
set -euo pipefail

# Start the server with the matching --agent_type before running this demo.
# Usage: bash script/run_client.sh <rag|kb_rag|rpg|deep_search> [base_url]
# Example: bash script/run_client.sh kb_rag http://localhost:30000

agent_type="${1:-}"
base_url="${2:-${BASE_URL:-http://localhost:30000}}"

send_request() {
    curl --fail-with-body --silent --show-error \
        --noproxy "localhost,127.0.0.1,::1,${NO_PROXY:-${no_proxy:-}}" \
        --connect-timeout 10 \
        --request POST "${base_url%/}/v1/chat" \
        --header 'Content-Type: application/json' \
        --data-binary @-
    printf '\n'
}

case "$agent_type" in
    rag)
        send_request <<'JSON'
{
  "agent_type": "rag",
  "query": "What is the capital of France?",
  "doc": "Paris is the capital of France. It is located on the Seine River.",
  "config": {
    "text_size": 1024,
    "top_k": 1,
    "rerank_top_k": 1
  }
}
JSON
        ;;
    rpg)
        send_request <<'JSON'
{
  "agent_type": "rpg",
  "query": "A red bicycle leaning against a blue wall in warm afternoon sunlight.",
  "doc": "",
  "config": {
    "n": 3
  }
}
JSON
        ;;
    kb_rag)
        # Build the global FAISS knowledge base before starting the server.
        send_request <<'JSON'
{
  "agent_type": "kb_rag",
  "query": "What is machine learning?",
  "doc": "",
  "config": {
    "top_k": 10,
    "rerank_top_k": 3
  }
}
JSON
        ;;
    deep_search)
        send_request <<'JSON'
{
  "agent_type": "deep_search",
  "query": "How do solar panels generate electricity?",
  "doc": "",
  "config": {
    "num_subagents": 3,
    "search_top_k": 3
  }
}
JSON
        ;;
    -h|--help|"")
        printf 'Usage: bash script/run_client.sh <rag|kb_rag|rpg|deep_search> [base_url]\n'
        printf 'Start a server with the matching agent type. Each run submits one request.\n'
        ;;
    *)
        printf 'Unknown agent type: %s\n' "$agent_type" >&2
        exit 2
        ;;
esac
