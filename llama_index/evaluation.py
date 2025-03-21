
from datasets import load_dataset
from rag import RAG
from tqdm import tqdm
from dotenv import load_dotenv
import logging
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage
import argparse
import json


logging.basicConfig(
    filename='llama_index_evaluation.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


parser = argparse.ArgumentParser(description='Evaluate a Q/A system using llm as judge.')
parser.add_argument('output_dir', type=str, help='The directory where the output will be saved.')
args = parser.parse_args()

load_dotenv()


if __name__ == "__main__":
    dataset_name = "huggingface_doc_qa_10_docs"
    ds = load_dataset(f"GTimothee/{dataset_name}")['train']
    rag_engine = RAG()

    llm = ChatOpenAI(
        openai_api_base=os.getenv("OPENAI_BASE_URL"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        timeout=60,
        request_timeout=60,
        model_name="Llama-3-70B-Instruct",
        temperature=0.0,
    )

    critique_chain = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                evaluation_prompt,
            ),
            ("placeholder", "{messages}"),
        ]
    ) | llm | JsonOutputParser()

    results = []

    for sample_idx, sample in tqdm(enumerate(ds), total=len(ds), desc="RAG evaluation running..."):
        logging.info(f"Processing sample {sample_idx}...")

        try:
            logging.info("- Running RAG engine...")
            rag_answer = rag_engine.run(sample['question'])
            logging.info(f"- RAG response: {rag_answer}")
            sample['rag_answer'] = rag_answer['text']
            sample['rag_context'] = rag_answer['context']
        except Exception as e:
            logging.error(f"Error processing sample {sample_idx}: {e}")
            sample['rag_answer'] = None
            sample['rag_context'] = None
            sample['rag_score'] = None
            sample['rag_feedback'] = None
            results.append(sample)
            continue

        try:
            logging.info("- Running critique chain...")
            critique = critique_chain.invoke(
                (
                    {"messages": [HumanMessage(content=f"Question:{sample['question']}\nExpected Answer:{sample['answer']}\nSystem's response:```{rag_answer}```")]})
            )
            logging.info(f"- Critique: {critique}")
            sample['rag_score'] = critique['evaluation']['score']
            sample['rag_feedback'] = critique['evaluation']['feedback']
        except Exception as e:
            logging.error(f"Error during evaluation of sample {sample_idx}: {e}")
            sample['rag_score'] = None
            sample['rag_feedback'] = None
        
        results.append({
            'sample_idx': sample_idx,
            'question': sample['question'],
            'answer': sample['answer'],
            'rag_answer': sample['rag_answer'],
            'rag_score': sample['rag_score'],
            'rag_feedback': sample['rag_feedback'],
            'rag_context': sample['rag_context']
        })

    output_path = os.path.join(args.output_dir, f'{dataset_name}_eval.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)