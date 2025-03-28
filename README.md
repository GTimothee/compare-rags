# Graph RAG comparative study

// Work in progress

No time to read the doc ? 
   You can already open the ```run_experiments.ipynb``` notebook in google colab and use the package with local LLM and embedding models!

## Introduction

Objectives of the repo:
- Compare python graph RAG frameworks
   - which one is the simplest to setup and use? 
   - which one gives the best results without too much tuning? 
   - what is the best performance I can expect with a given framework? What is the optimal configuration? 
   - How far in complexity can we go? For example can we add some graphML on top of our knowledge graph? Is it simple or complicated to setup? Is the performance gain worth the effort? 
- Learn the inner workings of graph RAG
   - Having implemented my own graphRAG solution with Noe4j python, I am also curious of how each solution has been implemented, and how they found solutions to the problems I faced.

What it gives you:
- An exemple of how to use each framework
- A comparative study to help you choose the best framework for your needs

Frameworks to test:
- [x] Llama-index
- [x] LightRAG
- [ ] Neo4j-graphrag

## How the study is designed

We provide a suite of scripts to generate a graph db and perform graphRAG on it, for each of the framework supported by this repo. We also have scripts to generate evaluation sets and evaluate the frameworks on them. 

Each experiment has a dedicated folder in experiments/{framework}/{version}, with a config file in it describing the parameters of the experiment (RAG config, source dataset of the documents in the DB, etc.). When you want to perform a new experiment, just create a new folder and a config file in it and run the different scripts in the scripts/ section, passing the config file.

- scripts/build_graph.py allows to generate a graph db (locally for now)
- scripts/test_rag.py allows to test the rag on the generated graph by passing a question as argument
- scritps/evaluation contains scripts to evaluate a rag config on a given evaluation set and analyze the results
- scripts/{dataset} contains scripts linked to a given source dataset, for example scripts to generate evaluation sets

## Running the scripts

0. Write a config file or use an existing one
1. Build the db ```python scripts/build_graph.py experiments/{framework}/test/config.yaml```
2. Test the RAG ```python scripts/test_rag.py experiments/{framework}/test/config.yaml "how to create an endpoint"``` (add ```--verbose``` to have a look at the context that has been retrieved)
3. Run evaluation:
   1. evaluate ```python scripts/evaluation/evaluate.py experiments/{framework}/out-of-the-box/config.yaml```
   2. analyze results ```python scripts/evaluation/analysis.py experiments/{framework}/out-of-the-box/eval.json```

### Example outputs for the test question "how to create an endpoint"

<details>
<summary>Answer from llama-index</summary>
To create an endpoint, follow these steps: 

1. Enter the required Hugging Face Repository ID and your desired endpoint name.
2. Select your Cloud Provider and region.
3. Define the Security Level for the Endpoint.
4. Create your Endpoint by clicking **Create Endpoint**. 
5. Wait for the Endpoint to build, initialize, and run, which can take between 1 to 5 minutes.
6. Test your Endpoint in the overview with the Inference widget.
</details>
<details>
<summary>Answer from lightRAG (top_k=3)</summary>
Creating an Endpoint on Hugging Face involves several steps, which are outlined below:

**Step 1: Enter the Hugging Face Repository ID and your desired endpoint name**

Go to the [Endpoint creation page](https://ui.endpoints.huggingface.co/new) and enter the Hugging Face Repository ID and your desired endpoint name.

**Step 2: Select your Cloud Provider and region**

Select your Cloud Provider (initially, only AWS is available) and region (either `us-east-1` or `eu-west-1`). You can also request to test Endpoints with other Cloud Providers or regions.

**Step 3: Define the Security Level for the Endpoint**

Define the Security Level for the Endpoint.

**Step 4: Create your Endpoint**

Click **Create Endpoint**. By default, your Endpoint is created with a medium CPU (2 x 4GB vCPUs with Intel Xeon Ice Lake). The cost estimate assumes the Endpoint will be up for an entire month and does not take autoscaling into account.

**Step 5: Wait for the Endpoint to build, initialize, and run**

Wait for the Endpoint to build, initialize, and run, which can take between 1 to 5 minutes.

**Step 6: Test your Endpoint**

Test your Endpoint in the overview with the Inference widget.

Here's an example of how to deploy the `distilbert-base-uncased-finetuned-sst-2-english` model for text classification:

<img src="https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/1_repository.png" alt="select repository" />
<img src="https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/1_region.png" alt="select region" />
<img src="https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/1_security.png" alt="define security" />
<img src="https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/1_create_cost.png" alt="create endpoint" />
<img src="https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/overview.png" alt="overview" />
<img src="https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/1_inference.png" alt="run inference" />

References:
[1. DC] Create an Endpoint
[2. KG] Hugging Face, organization
[3. KG] Endpoint creation page, category
</details>

## Results

On the ["huggingface 10 docs"](https://huggingface.co/datasets/GTimothee/huggingface_doc_qa_10_docs) evaluation dataset:

| Framework | Config | Version description |Normalized accuracy |
| --- | --- | --- | --- |
| llama-index | Out-of-the-box | Does not use embeddings (default behaviour without using a graphDB) | 74.7% |
| lightrag | Out-of-the-box | / | 81% |

Frameworks usage: 
- lightRAG
   - Documentation is raw and sparse
   - By default it is retrieving LOTS of data to answer a query
   - Nice that they add source citation by default
- llama-index
   - Documentation is good
   - API is nice, providing a good balance between abstraction and customization
   - I appreciate the fact that we can easily see what we are doing and customize. It is very lightweight.
   - Sad that we are forced to use a graphDB to be able to do vector search for graphRAG. I am wondering how lightRAG does, as it seems to be using vector search without it. Apparently it is possible to use a standard vector DB to store the nodes and use a separate graphDB for the graph search part, but it will take a bit of time (which I don't have). 

## Source datasets

### Huggingface docs
- Link: https://huggingface.co/datasets/m-ric/huggingface_doc
- Source: https://huggingface.co/learn/cookbook/en/rag_evaluation
- The idea of the above cookbook was not bad at all: 1) generate db containing data from a 1000 documents and 2) generate a few questions from random chunks in the corpus. In a spirit of saving tokens and easier debugging, I decided to only process a few documents, and to evaluate only on that. That is why I generated a new test set. 
- Generated evaluation dataset from the first 10 documents of the dataset (139 questions/answers): https://huggingface.co/datasets/GTimothee/huggingface_doc_qa_10_docs

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

