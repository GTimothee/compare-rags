import os
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core import Document
from src.configuration import Config
from src.llama_index.models import get_llm, get_embedding_model
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser, MarkdownNodeParser
from llama_index.core.extractors import (
    SummaryExtractor,
    TitleExtractor,
    KeywordExtractor,
    QuestionsAnsweredExtractor
)
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore


class LlamaIndexGraphBuilder:
    def __init__(self, config: Config):
        self.config = config
        self.llm = get_llm(config.llm)
        self.embed_model = get_embedding_model(config.embedding_model)
        self.index_dirpath = config.index_dirpath

        if config.graph_store == 'neo4j':
            self.graph_store = Neo4jPropertyGraphStore(
                username=os.getenv('NEO4J_USERNAME'),
                password=os.getenv('NEO4J_PASSWORD'),
                url=os.getenv('NEO4J_URL'),
                database=os.getenv('NEO4J_DATABASE'),
            )
        else:
            self.graph_store = SimplePropertyGraphStore()

    def build(self, ds):

        docs = [Document(text=sample['text']) for sample in ds]
        
        match self.config.version:
            case 'v1':
                kg_extractors, transformations, include_embeddings = self._version_v1()
            case _:
                kg_extractors, transformations, include_embeddings = self._default_version()
                print("Using default version.")
        
        index = PropertyGraphIndex.from_documents(
            docs, 
            kg_extractors=kg_extractors, 
            embed_model=self.embed_model,
            show_progress=True,
            transformations=transformations,
            max_triplets_per_chunk=20, # the default
            include_embeddings=include_embeddings,
            property_graph_store=self.graph_store,
        )
        
        print('Saving the index to disk...')
        index.storage_context.persist(persist_dir=str(self.index_dirpath))

    def _default_version(self):
        kg_extractors = [SimpleLLMPathExtractor(llm=self.llm)]
        transformations = None
        include_embeddings = False
        return kg_extractors, transformations, include_embeddings
    
    def _version_v1(self): # TODO change store
        kg_extractors = [SimpleLLMPathExtractor(llm=self.llm)]
        transformations = [
            SentenceSplitter(
                chunk_size = 1024, # default
                chunk_overlap = 200,
                paragraph_separator = '\n\n'
            ),
            TitleExtractor(nodes=5, llm=self.llm),
            QuestionsAnsweredExtractor(questions=3, llm=self.llm),
            SummaryExtractor(summaries=["prev", "self"], llm=self.llm),
            KeywordExtractor(keywords=10, llm=self.llm)
        ]
        include_embeddings = True
        return kg_extractors, transformations, include_embeddings
    
    # def _version_v2(self):
    # DynamicLLMPathExtractor?
    # EntityExtractor ?
    #     kg_extractors = [SimpleLLMPathExtractor(llm=self.llm)]
    #     transformations = [SemanticSplitterNodeParser()]
    #     return kg_extractors, transformations
    
    # def _version_v3(self):
    #     kg_extractors = [SimpleLLMPathExtractor(llm=self.llm)]
    #     transformations = [MarkdownNodeParser(
    #         include_metadata = True,
    #         include_prev_next_rel = True
    #     )]
    #     return kg_extractors, transformations