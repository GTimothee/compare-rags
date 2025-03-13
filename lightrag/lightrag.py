"""
pip install lightrag-hku
"""

import os
import asyncio

import tqdm
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger
import datasets
from lightrag.llm.hf import hf_embed
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

setup_logger("lightrag", level="INFO")


hf_embeddings = EmbeddingFunc(
    embedding_dim=384,
    max_token_size=5000,
    func=lambda texts: hf_embed(
        texts,
        tokenizer=AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        embed_model=AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
    ),
)


async def initialize_rag():

    rag = LightRAG(
        working_dir="your/path",
        embedding_func=openai_embed, # hf_embeddings
        llm_model_func=gpt_4o_mini_complete
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag  


if __name__ == "__main__":
    rag = asyncio.run(initialize_rag())
    
    ds = datasets.load_dataset("m-ric/huggingface_doc", split="train").select(range(30))

    for index, doc in enumerate(tqdm(ds, desc="Processing documents")):
        rag.insert(doc["text"])

    output = rag.query(
        "What is SQuAD?",
        param=QueryParam(mode="mix")
    )
    print(output)

    output = rag.query(
        "What is the default CPU configuration for a created endpoint?",
        param=QueryParam(mode="mix")
    )
    print(output)
