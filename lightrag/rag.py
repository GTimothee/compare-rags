"""
pip install lightrag-hku
"""

import os
import asyncio
import numpy as np
from tqdm import tqdm
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed, openai_complete_if_cache
# from lightrag.kg.shared_storage import initialize_pipeline_status
# from lightrag.utils import setup_logger
import datasets
from lightrag.llm.hf import hf_embed
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer
import time
from dotenv import load_dotenv

load_dotenv()

# setup_logger("lightrag", level="INFO")


# hf_embeddings = EmbeddingFunc(
#     embedding_dim=384,
#     max_token_size=5000,
#     func=lambda texts: hf_embed(
#         texts,
#         tokenizer=AutoTokenizer.from_pretrained(
#             "sentence-transformers/all-MiniLM-L6-v2"
#         ),
#         embed_model=AutoModel.from_pretrained(
#             "sentence-transformers/all-MiniLM-L6-v2"
#         ),
#     ),
# )



async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "Llama-3-70B-Instruct",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        **kwargs
    )

async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model="gte-large-en-v1.5",
        api_key=os.getenv("embedding_OPENAI_API_KEY"),
        base_url=os.getenv("embedding_OPENAI_BASE_URL")
    )


async def initialize_rag():

    rag = LightRAG(
        # working_dir="your/path",
        embedding_func=EmbeddingFunc(
            embedding_dim=4096,
            max_token_size=8192,
            func=embedding_func
        ), #openai_embed, # hf_embeddings
        llm_model_func=llm_model_func
    )

    await rag.initialize_storages()
    # await initialize_pipeline_status()

    return rag  


if __name__ == "__main__":
    rag = asyncio.run(initialize_rag())
    
    ds = datasets.load_dataset("m-ric/huggingface_doc", split="train").select(range(8))

    t = time.time()
    for index, doc in enumerate(tqdm(ds, desc="Processing documents")):
        rag.insert(doc["text"])
    ingestion_time = time.time() - t
    print(f'Ingestion time: {ingestion_time:.2f} seconds.')

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
