from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core import Document
from src.llama_index.models import llm as base_llm, embed_model as base_embed_model
        

class LlamaIndexGraphBuilder:

    def __init__(self, index_dirpath, llm=base_llm, embed_model=base_embed_model):
        self.llm = llm
        self.embed_model = embed_model
        self.index_dirpath = index_dirpath

    def build(self, ds):
        docs = [Document(text=sample['text']) for sample in ds]
        kg_extractor = SimpleLLMPathExtractor(llm=self.llm)
        
        index = PropertyGraphIndex.from_documents(
            docs, 
            kg_extractors=[kg_extractor], 
            embed_model=self.embed_model,
            show_progress=True)
        
        print('Saving the index to disk...')
        index.storage_context.persist(persist_dir=str(self.index_dirpath))