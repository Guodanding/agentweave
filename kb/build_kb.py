import argparse
import os
import sys
import time
import torch
import threading
import queue
import json
import logging
from collections import OrderedDict
from typing import Dict, List, Any, Tuple, Optional, Union
import pickle
import numpy as np
import faiss

# =================================================================
# =================================================================


try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import TextNode, NodeWithScore
    from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from pydantic import PrivateAttr
    from datasets import load_dataset
    import sglang as sgl
    import math
except ImportError as e:
    logger.error(f"Missing required dependency: {e}. Run 'pip install transformers sentence-transformers accelerate llama-index pydantic torch' to install dependencies.")
    exit()

logger = logging.getLogger(__name__)

class Embedding:
    def __init__(self, model_path: str, mem_fraction_static: float = 0.8):
        self.engine = sgl.Engine(model_path=model_path, tp_size=4, is_embedding=True, allow_auto_truncate=True, mem_fraction_static=mem_fraction_static)#, base_gpu_id=1, port=30001)

    def encode(
        self,
        prompts: Union[str, List[str]],
    ) -> Union[List[float], List[List[float]]]:
        if not prompts:
            return []
        is_single_prompt = isinstance(prompts, str)
        prompts = [prompts] if is_single_prompt else prompts
        outputs = self.engine.encode(prompts)
        embeddings = [output['embedding'] for output in outputs]
        return embeddings[0] if is_single_prompt else embeddings
    
    def shutdown(self):
        print("Shutting down SGLang Embedding Engine...")
        self.engine.shutdown()

def build_global_knowledge_base(
    dataset: List[Dict], 
    embed_client: Embedding, 
    chunk_size: int = 1024,
    index_path: str = "global_kb.faiss", 
    meta_path: str = "global_kb_meta.pkl",
    overwrite_cache: bool = False,
    num_samples: int = 100,
):
    if os.path.exists(index_path) and os.path.exists(meta_path) and not overwrite_cache:
        print(f"Local knowledge base cache found ({index_path}). Specify a different path...")
        return None, None

    print("No local knowledge base found. Starting dataset preprocessing and embedding generation...")
    text_parser = SentenceSplitter(chunk_size=chunk_size)
    all_nodes = []

    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        doc_text = "\n\n".join(item['search_results']['search_context']) + "\n"
        chunks = text_parser.split_text(doc_text)
        for chunk in chunks:
            node = TextNode(text=chunk, metadata={"source_dataset_idx": i})
            all_nodes.append(node)
            
    print(f"Document splitting complete: {len(all_nodes)} chunks created. Starting batch embedding generation...")

    batch_size = 128
    all_embeddings = []
    
    for i in range(0, len(all_nodes), batch_size):
        batch_nodes = all_nodes[i:i+batch_size]
        batch_texts = [n.get_content() for n in batch_nodes]
        embeddings = embed_client.encode(batch_texts)
        all_embeddings.extend(embeddings)
        if i % (batch_size * 10) == 0:
            print(f"Preprocessing progress: {i}/{len(all_nodes)}")

    embedding_dim = len(all_embeddings[0])
    print(f"Building FAISS index with embedding dimension: {embedding_dim}")
    
    embeddings_np = np.array(all_embeddings).astype('float32')
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)

    nodes_mapping = {idx: node for idx, node in enumerate(all_nodes)}

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(nodes_mapping, f)

    print("Global knowledge base built and saved successfully!")
    return index, nodes_mapping


def main():
    parser = argparse.ArgumentParser(description="Build FAISS Knowledge Base with SGLang Embeddings")

    parser.add_argument("--model_path", type=str, required=True, 
                        help="Embedding model path (local path or Hugging Face model ID)")
    parser.add_argument("--dataset_path", type=str, required=True, 
                        help="Dataset path (JSON, JSONL, or Hugging Face dataset name)")

    parser.add_argument("--index_path", type=str, default="global_kb.faiss", 
                        help="Output path for the FAISS index")
    parser.add_argument("--meta_path", type=str, default="global_kb_meta.pkl", 
                        help="Output path for metadata (TextNodes)")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Text chunk size")
    parser.add_argument("--overwrite_cache", action="store_true", 
                        help="Overwrite existing index files and rebuild")
    parser.add_argument("--mem_fraction", type=float, default=0.8, 
                        help="SGLang GPU memory fraction (0.1 - 1.0)")
    parser.add_argument("--num_samples", type=int, default=100, 
                        help="Process only the first N records (for debugging)")

    args = parser.parse_args()
    
    print("=== Starting knowledge base build ===")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output Index: {args.index_path}")

    if os.path.exists(args.index_path) and os.path.exists(args.meta_path) and not args.overwrite_cache:
        print(f"Target file {args.index_path} already exists and overwrite_cache is not enabled.")
        print("Exiting without rebuilding.")
        sys.exit(0)

    dataset = load_dataset(args.dataset_path, "rc", split='validation')

    embed_client = None
    try:
        embed_client = Embedding(
            model_path=args.model_path, 
            mem_fraction_static=args.mem_fraction
        )
        
        build_global_knowledge_base(
            dataset=dataset,
            embed_client=embed_client,
            chunk_size=args.chunk_size,
            index_path=args.index_path,
            meta_path=args.meta_path,
            overwrite_cache=args.overwrite_cache,
            num_samples=args.num_samples
        )

    except KeyboardInterrupt:
        print("Task interrupted by user.")
    except Exception as e:
        logger.error(f"Unhandled runtime exception: {e}", exc_info=True)
    finally:
        if embed_client:
            embed_client.shutdown()
        print("=== Task finished ===")

if __name__ == "__main__":
    main()
