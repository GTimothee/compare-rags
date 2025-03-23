from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
import datasets
import logging
import os
from langchain_openai import ChatOpenAI, OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import sys
from langchain_core.messages import HumanMessage, SystemMessage
import argparse
from dotenv import load_dotenv
import json


load_dotenv()


logging.basicConfig(
    filename='evaluation.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


parser = argparse.ArgumentParser(description='Generate QA pairs and save to an output directory.')
parser.add_argument('output_dir', type=str, help='The directory where the QA pairs will be saved.')
args = parser.parse_args()


QA_generation_prompt = """Extract a list of fact-based question-answer pairs from the given text. Each question must be standalone, meaning it must be fully understandable without referring to the original passage.
Rules for Questions:

    The question must include all necessary context to be understood independently.
    The question must be fact-based and refer to a specific detail in the text. Avoid vague references like ‘the overview’ or ‘the passage.’
    The question should resemble a search engine query (e.g., ‘Who discovered gravity?’ instead of ‘Who is mentioned in the passage?’).
    DO NOT use placeholders, ambiguous wording, or subjective interpretations.

Rules for Answers:

    The answer must be a concise, factual sentence derived directly from the text.
    The answer must be specific (avoid generalizations or assumptions).

Output Format (JSON only):
{{
    [
        {{
            "question": (your factoid question),
            "answer": (your answer to the factoid question)
        }},
        ...
    ]
}}

Do not include any additional text—only the JSON object. Any extra content will result in a grade of 0."""

combined_critique_prompt = """
You will be given a context along with a question and answer pair. Your task is to provide three total ratings based on the following criteria:
1. Groundedness (1-5)

    5: The question is clearly and unambiguously answerable with the provided context. No ambiguity or missing details.
    4: The question is answerable, but minor inference is needed.
    3: The answer can be inferred, but some details are unclear.
    2: The question is only partially answerable or too vague.
    1: The question cannot be answered at all using the given context.

2. Standalone Clarity (1-5)

    5: The question is fully independent and understandable on its own. No reference to "the passage," "this context," etc.
    4: The question is mostly standalone but could be slightly clearer.
    3: Some missing context makes it harder to understand.
    2: The question assumes external information not given.
    1: The question is completely dependent on context references and is unclear.

3. Answer Consistency (1-5)

    5: The answer is direct, correct, and explicitly found in the given context.
    4: The answer is correct but could be more precise.
    3: The answer is somewhat relevant but may be incomplete.
    2: The answer is unclear or only partially addresses the question.
    1: The answer does not match the question or is incorrect.

Final Instructions:

    Consider practical usability instead of being overly strict. If a reasonable human could understand and use the question-answer pair correctly, avoid penalizing too harshly.
    Provide your answer as a JSON object with the following structure:

{{
    "groundedness": {{
        "evaluation": "(your rationale for the rating, as a text)",
        "total_rating": (your rating, as a number between 1 and 5)
    }},
    "standalone": {{
        "evaluation": "(your rationale for the rating, as a text)",
        "total_rating": (your rating, as a number between 1 and 5)
    }},
    "consistency": {{
        "evaluation": "(your rationale for the rating, as a text)",
        "total_rating": (your rating, as a number between 1 and 5)
    }}
}}

You MUST provide values for 'evaluation' and 'total_rating' for each criterion in your answer.

IMPORTANT: Only output the JSON object, nothing else. Do not add any additional information or context. Failure to comply will incur a grade of 0.
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

    total_pairs_generated = 0
    total_pairs_kept = 0

    for chunk in tqdm(chunks):

        # generate pairs
        try:
            logging.info(f"Generating QA pairs for document {chunk.metadata['source']}...")
            qa_pairs_list = generation_chain.invoke(
                {"messages": [HumanMessage(content=f"Here is the context:```{chunk.page_content}```")]})
        except Exception as e:
            logging.error(f"Error while generating qa pair: {e}")
            continue

        try:
            assert qa_pairs_list
            assert isinstance(qa_pairs_list, list)
            assert len(qa_pairs_list)
        except Exception as e:
                logging.error(f"Error while parsing generation output: {e}")
                continue
        
        total_pairs_generated += len(qa_pairs_list)
        logging.info(f"Generated {len(qa_pairs_list)} QA pairs for document {chunk.metadata['source']}")

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
                    (
                        {"messages": [HumanMessage(content=f"Question:{question}\nAnswer:{answer}\nContext:```{chunk.page_content}```")]})
                )
            except Exception as e:
                logging.error(f"Error while generating critique: {e}")
                continue
            
            # parse critique
            try:
                logging.info(f"- Parsing critique output: {critique}")
                groundedness = critique['groundedness']
                standalone = critique['standalone']
                consistency = critique['consistency']

                assert groundedness['total_rating'] >= 4, f"Question is not grounded (score={groundedness['evaluation']})"
                assert standalone['total_rating'] >= 4, f"Question is not standalone (score={standalone['evaluation']})"
                assert consistency['total_rating'] >= 4, f"Answer is not consistent (score={consistency['evaluation']})"

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

            total_pairs_kept += 1
            
    logging.info(f"Total QA pairs generated: {total_pairs_generated}")
    logging.info(f"Total QA pairs kept: {total_pairs_kept}")
    print(f"Total QA pairs generated: {total_pairs_generated}")
    print(f"Total QA pairs kept: {total_pairs_kept}")
    if total_pairs_generated > 0:
        ratio = total_pairs_kept / total_pairs_generated
    else:
        ratio = 0
    logging.info(f"Ratio of QA pairs kept: {ratio:.2f}")
    print(f"Ratio of QA pairs kept: {ratio:.2f}")

    return outputs


if __name__ == "__main__":

    N_DOCS = 10

    print('Loading dataset...')
    ds = datasets.load_dataset("m-ric/huggingface_doc", split="train").select(range(N_DOCS))

    print('Formatting documents...')
    langchain_docs = [
        LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(ds)]

    print('Chunking documents...')
    chunks = chunk_data(langchain_docs)

    print('Generating QA pairs...')
    llm = ChatOpenAI(
        openai_api_base=os.getenv("OPENAI_BASE_URL"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        timeout=60,
        request_timeout=60,
        model_name="Llama-3-70B-Instruct",
        temperature=0.0,
    )

    generation_chain = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    QA_generation_prompt,
                ),
                ("placeholder", "{messages}"),
            ]
        ) | llm | JsonOutputParser()
    
    critique_chain = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    combined_critique_prompt,
                ),
                ("placeholder", "{messages}"),
            ]
        ) | llm | JsonOutputParser()
    
    qa_pairs = generate_qa_pairs(generation_chain, critique_chain, chunks)
    
    print('Number of QA pairs:', len(qa_pairs))
    print('Saving QA pairs...')
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'qa_pairs_{N_DOCS}.json')

    with open(output_path, 'w') as f:
        json.dump(qa_pairs, f, indent=4)
