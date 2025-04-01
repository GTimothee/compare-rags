import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.evaluation.prompts import evaluation_prompt


class OpenAILikeEvaluator:
    def __init__(self, model_name: str = "Llama-3-70B-Instruct"):
        self.llm = ChatOpenAI(
            openai_api_base=os.getenv("OPENAI_BASE_URL"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            timeout=60,
            request_timeout=60,
            model_name=model_name,
            temperature=0.0,
        )
        self.critique_chain = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    evaluation_prompt,
                ),
                ("human", "{user_message}"),
            ]
        ) | self.llm | JsonOutputParser()

    def evaluate(self, user_message: str):
        return self.critique_chain.invoke({
                "user_message": user_message
            }
        )