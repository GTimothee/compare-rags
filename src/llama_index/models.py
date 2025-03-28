"""Any llm or embedding model implementations compatible with llama-index interfaces.
"""

import os 
from langchain_openai import OpenAIEmbeddings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.mistralai  import MistralAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import set_global_tokenizer
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def get_llm(llm_name: str):
    if llm_name.startswith("huggingface-api"):
        _, model_name = llm_name.split("_")
        set_global_tokenizer(
            AutoTokenizer.from_pretrained(model_name).encode
        )
        return HuggingFaceInferenceAPI(
            model_name=model_name, token=os.getenv('HF_TOKEN')
        )
    elif llm_name.startswith("huggingface"):
        _, model_name = llm_name.split("_")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
        return HuggingFaceLLM(model=model, tokenizer=tokenizer)
    elif llm_name == "openai-like":
        return OpenAILike(
            model="Llama-3-70B-Instruct", 
            is_chat_model=True, 
            is_function_calling_model=False,
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_BASE_URL")
        )
    elif llm_name == "mistral": 
        return MistralAI(model="mistral-large-latest", temperature=0.3)
    else:
        raise NotImplementedError(f"Unsupported llm: {llm_name}")


def get_embedding_model(model_name: str):
    if model_name == "openai-like":
        return LangchainEmbedding(
            langchain_embeddings=OpenAIEmbeddings(
                model="gte-large-en-v1.5",
                api_key=os.getenv("embedding_OPENAI_API_KEY"),
                base_url=os.getenv("embedding_OPENAI_BASE_URL")
            )
        )
    elif model_name == "huggingface-sentence-transformers":
        return HuggingFaceEmbedding("all-MiniLM-L6-v2")
    else:
        raise NotImplementedError(f"Unsupported embedding model: {model_name}")