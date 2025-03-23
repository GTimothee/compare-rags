"""Any llm or embedding model implementations compatible with llama-index interfaces.
"""

import os 
from langchain_openai import OpenAIEmbeddings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.mistral import Mistral
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer

# <<<<<<<< llms >>>>>>>>

llm = OpenAILike(
    model="Llama-3-70B-Instruct", 
    is_chat_model=True, 
    is_function_calling_model=False,
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"))

mistral_llm = Mistral(model="mistral-large-latest", temperature=0.3)

# <<<<<<<< embed models >>>>>>>>

embed_model = LangchainEmbedding(
    langchain_embeddings=OpenAIEmbeddings(
        # dimensions=1024,
        model="gte-large-en-v1.5",
        api_key=os.getenv("embedding_OPENAI_API_KEY"),
        base_url=os.getenv("embedding_OPENAI_BASE_URL")
        # timeout=500,
    )
)

st_model = SentenceTransformer("all-MiniLM-L6-v2")
st_embed_model = HuggingFaceEmbedding(model=st_model)