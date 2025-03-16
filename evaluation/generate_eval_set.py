from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
import datasets
import logging
import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.prompt_values import PromptValue
from langchain_core.output_parsers import JsonOutputParser
import sys
import argparse


logging.basicConfig(
    filename='evaluation.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


parser = argparse.ArgumentParser(description='Generate QA pairs and save to an output directory.')
parser.add_argument('output_dir', type=str, help='The directory where the QA pairs will be saved.')
args = parser.parse_args()


QA_generation_prompt = """You are a helpful assistant.
Your task is to write a list of factoid question/answer pairs, given a context.

Generation rules for the factoid questions:
- Each factoid question should be answerable with a specific, concise piece of factual information from the context.
- Each factoid question should be formulated in the same style as questions users could ask in a search engine. This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Generation rules for the answers:
- Each answer should be a concise piece of factual information from the context that answers the corresponding factoid question.
- Each answer should be a single sentence.

Provide your answer as a JSON object with key "data", containing a list of dictionaries, where each dictionary has two keys: "question" and "answer".

Example:
{
    "data": [
        {
            "question": (your factoid question),
            "answer": (your answer to the factoid question)
        },
        ...
    ]
}

IMPORTANT: Only output the JSON object, nothing else. Do not add any additional information or context. Failure to comply will incur a grade of 0.

Now here is the context:
{context}
"""

combined_critique_prompt = """
You will be given a context and a question.
Your task is to provide three 'total ratings' based on the following criteria:

1. Groundedness: How well one can answer the given question unambiguously with the given context.
    Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

2. Standalone: How context-independent this question is.
    Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
    For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
    The questions can contain obscure technical nouns or acronyms like Gradio, Hub, Hugging Face or Space and still be a 5: it must simply be clear to an operator with access to documentation what the question is about.
    For instance, "What is the name of the checkpoint from which the ViT model is imported?" should receive a 1, since there is an implicit mention of a context, thus the question is not independent from the context.

Give your answers on a scale of 1 to 5, where 1 is the lowest rating and 5 is the highest rating.

Provide your answer as a JSON object with the following structure:
{
    "groundedness": {
        "evaluation": "(your rationale for the rating, as a text)",
        "total_rating": (your rating, as a number between 1 and 5)
    },
    "standalone": {
        "evaluation": "(your rationale for the rating, as a text)",
        "total_rating": (your rating, as a number between 1 and 5)
    }
}

You MUST provide values for 'evaluation' and 'total_rating' for each criterion in your answer.

IMPORTANT: Only output the JSON object, nothing else. Do not add any additional information or context. Failure to comply will incur a grade of 0.

Now here are the question and context.

Question: {question}\n
Context: {context}\n
"""


def chunk_data(langchain_docs: list[LangchainDocument]) -> list[LangchainDocument]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    docs_processed = []
    for doc in langchain_docs:
        docs_processed += text_splitter.split_documents([doc])

    return docs_processed


def generate_qa_pairs(
        generation_chain, 
        critique_chain, 
        chunks: list[LangchainDocument]
    ) -> list[LangchainDocument]:

    outputs = []
    logging.info("Generating QA pairs from chunks...")

    for chunk in tqdm(chunks):

        # generate pairs
        logging.info(f"Generating QA pairs for document {chunk.metadata['source']}...")
        output_QA_couple = generation_chain.invoke(PromptValue({"context": chunk.page_content}))

        try:
            logging.info(f"- Parsing generation output: {output_QA_couple}")
            qa_pairs_list = output_QA_couple["data"]
        except Exception as e:
            logging.error(f"Error while parsing the generation output: {e}")
            continue
        
        # for each pair
        for qa_pair in qa_pairs_list:

            # parse
            try:
                logging.info(f"- Parsing QA couple: {qa_pair}")
                question = qa_pair["question"]
                answer = qa_pair["answer"]
                assert len(answer) < 300, "Answer is too long"
                
            except Exception as e:
                logging.error(f"Error while parsing QA couple: {e}")
                continue

            # get critique
            try:
                logging.info(f"- Generating critique for question: {question}")
                critique = critique_chain.invoke(
                    {
                        "question": question,
                        "context": answer,
                    }
                )
            except Exception as e:
                logging.error(f"Error while generating critique: {e}")
                continue
            
            # parse critique
            try:
                logging.info(f"- Parsing critique output: {critique}")
                groundedness = critique['groundedness']
                standalone = critique['standalone']

                assert groundedness['total_rating'] >= 4, f"Question is not grounded (score={groundedness['evaluation']})"
                assert standalone['total_rating'] >= 4, f"Question is not standalone (score={standalone['evaluation']})"

            except Exception as e:
                logging.error(f"Error while evaluating the critique output: {e}")
                continue

            # critique passed -> append to outputs
            logging.info(f"- QA couple passed critique: {question}")
            outputs.append(
                {
                    "context": chunk.page_content,
                    "question": question,
                    "answer": answer,
                    "source_doc": chunk.metadata["source"],
                }
            )
        
    return outputs


if __name__ == "__main__":

    N_DOCS = 30

    print('Loading dataset...')
    ds = datasets.load_dataset("m-ric/huggingface_doc", split="train").select(range(N_DOCS))

    print('Formatting documents...')
    langchain_docs = [
        LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(ds)]

    print('Chunking documents...')
    chunks = chunk_data(langchain_docs)

    print('Generating QA pairs...')
    llm = OpenAI(
        openai_api_base=os.getenv("OPENAI_BASE_URL"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        timeout=60,
        request_timeout=60,
        model_name="Llama-3-70B-Instruct",
        temperature=0.0,
    )
    generation_chain = llm | PromptTemplate.from_template(QA_generation_prompt) | JsonOutputParser()
    critique_chain = llm | PromptTemplate.from_template(combined_critique_prompt) | JsonOutputParser()
    qa_pairs = generate_qa_pairs(generation_chain, critique_chain, chunks)
    
    print('Number of QA pairs:', len(qa_pairs))
    print('Saving QA pairs...')
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'qa_pairs.json')

    with open(output_path, 'w') as f:
        f.write(str(qa_pairs))
