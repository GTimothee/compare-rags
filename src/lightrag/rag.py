import os
import asyncio
import numpy as np
from tqdm import tqdm
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_embed, openai_complete_if_cache
from lightrag.utils import EmbeddingFunc


class LightRag:

    def __init__(self, working_dir):
        self.working_dir = working_dir
        self.rag = asyncio.run(self.initialize_rag())
        
    def build(self, ds):
        for _, doc in enumerate(tqdm(ds, desc="Processing documents")):
            self.rag.insert(doc["text"])

    def run(self, question: str, mode="mix") -> dict:
        answer = self.rag.query(
            question,
            param=QueryParam(mode=mode, top_k=3)
        )
        return {
            "text": answer,
            "context": None
        }

    @staticmethod
    async def llm_model_func(
        prompt, system_prompt=None, history_messages=[], **kwargs
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

    @staticmethod
    async def embedding_func(texts: list[str]) -> np.ndarray:
        return await openai_embed(
            texts,
            model="gte-large-en-v1.5",
            api_key=os.getenv("embedding_OPENAI_API_KEY"),
            base_url=os.getenv("embedding_OPENAI_BASE_URL")
        )

    async def initialize_rag(self):
        rag = LightRAG(
            working_dir=self.working_dir,
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=self.embedding_func
            ),
            llm_model_func=self.llm_model_func,
            llm_model_max_token_size=8000
        )
        await rag.initialize_storages()
        return rag  