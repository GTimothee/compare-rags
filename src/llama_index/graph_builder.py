from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core import Document
from src.llama_index.models import get_llm, get_embedding_model
        

class LlamaIndexGraphBuilder:

    def __init__(self, index_dirpath, llm_name, embed_model_name):
        self.llm = get_llm(llm_name)
        self.embed_model = get_embedding_model(embed_model_name)
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