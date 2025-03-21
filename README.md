# RAG comparative study

// Work in progress

## Comparing graphRAG systems

In progress.

### Llama-index

### LightRAG

### Neo4j-graphrag

### Results

| Framework | Config | Normalized accuracy |
| --- | --- | --- |
| llama-index | Out-of-the-box | 74.7% |

## Comparing RAG systems

TODO 

## Comparing RAG vs graphRAG

TODO

## Evaluation datasets

Huggingface doc':
- Link: https://huggingface.co/datasets/m-ric/huggingface_doc
- Source: https://huggingface.co/learn/cookbook/en/rag_evaluation
- The idea of the above cookbook was not bad at all: 1) generate db containing data from a 1000 documents and 2) generate a few questions from random chunks in the corpus. In a spirit of saving tokens and easier debugging, I decided to only process a few documents, and to evaluate only on that. That is why I generated a new test set. 
- Generated evaluation dataset from the first 10 documents of the dataset: https://huggingface.co/datasets/GTimothee/huggingface_doc_qa_10_docs

## Sources

Evaluation built based on the following cookbook by huggingface: https://huggingface.co/learn/cookbook/en/rag_evaluation

