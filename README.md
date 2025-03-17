# compare-rags

## Comparing RAG vs graphRAG

## Comparing graphRAG systems

### Datasets

Generated dataset: https://huggingface.co/datasets/GTimothee/huggingface_doc_qa_10_docs

## Comparing RAG systems

## Sources

Comparative study built based on the following cookbook by huggingface: https://huggingface.co/learn/cookbook/en/rag_evaluation

The idea of the above cookbook was not bad at all: 
1. generate db containing data from a 1000 documents
2. generate a few questions from random chunks in the corpus

In a spirit of saving tokens and easier debugging, I decided to only process a few documents, and to evaluate only on that. That is why I generated a new test set. 
