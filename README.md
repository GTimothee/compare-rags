# RAG comparative study

// Work in progress

Working with RAG, I was wondering what was the faster way to setup a light graphRAG, which framework to use and which of the main frameworks was the best. 

The goal of this repo is to compare the inner workings and overall performance of:
- Llama-index
- LightRAG
- Neo4j-graphrag

Having implemented my own graphRAG solution with Noe4j python, I am also curious of how each solution has been implemented, and how they found solutions to many problems I faced.

## How the study is designed

We provide a suite of scripts to generate a graph db and perform graphRAG on it, for each of the framework supported by this repo. We also have scripts to generate evaluation sets and evaluate the frameworks on them. 

Each experiment has a dedicated folder in experiments/{framework}/{version}, with a config file in it describing the parameters of the experiment (RAG config, source dataset of the documents in the DB, etc.). When you want to perform a new experiment, just create a new folder and a config file in it and run the different scripts in the scripts/ section, passing the config file.

- scripts/build_graph.py allows to generate a graph db (locally for now)
- scripts/test_rag.py allows to test the rag on the generated graph by passing a question as argument
- scritps/evaluation contains scripts to evaluate a rag config on a given evaluation set and analyze the results
- scripts/{dataset} contains scripts linked to a given source dataset, for example scripts to generate evaluation sets

## Running the scripts

1. build the db ```python scripts/build_graph.py experiments/llama_index/test/config.yaml```

## Results

On the ["huggingface 10 docs"](https://huggingface.co/datasets/GTimothee/huggingface_doc_qa_10_docs) evaluation dataset:

| Framework | Config | Normalized accuracy |
| --- | --- | --- |
| llama-index | Out-of-the-box | 74.7% |

## Source datasets

### Huggingface docs
- Link: https://huggingface.co/datasets/m-ric/huggingface_doc
- Source: https://huggingface.co/learn/cookbook/en/rag_evaluation
- The idea of the above cookbook was not bad at all: 1) generate db containing data from a 1000 documents and 2) generate a few questions from random chunks in the corpus. In a spirit of saving tokens and easier debugging, I decided to only process a few documents, and to evaluate only on that. That is why I generated a new test set. 
- Generated evaluation dataset from the first 10 documents of the dataset: https://huggingface.co/datasets/GTimothee/huggingface_doc_qa_10_docs

### MultiHopRAG
// TODO
- "yixuantt/MultiHopRAG"

train set: 
```
corpus = load_dataset("yixuantt/MultiHopRAG", "corpus")["train"]
docs = corpus["document"]
```

## Sources

Evaluation built based on the following cookbook by huggingface: https://huggingface.co/learn/cookbook/en/rag_evaluation

