python build_kb.py \
    --model_path /workspace/models/Qwen3-Embedding-8B \
    --dataset_path /workspace/datasets/trivalqa \
    --chunk_size 1024 \
    --index_path ./my_kb.faiss \
    --meta_path ./my_kb_meta.pkl \
    --overwrite_cache \
    --num_samples 1000