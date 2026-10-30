from typing import List
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode


def split_text_by_text_size(text: str, text_size: int) -> list[str]:
    text_parser = SentenceSplitter(chunk_size=text_size)
    text_chunks = text_parser.split_text(text)
    return text_chunks


def dispatch_text_into_nodes(text_chunks: list[str], request_id) -> list[TextNode]:
    nodes = []
    for chunk in text_chunks:
        node = TextNode(text=chunk, metadata={"request_id": request_id})
        nodes.append(node)
    return nodes