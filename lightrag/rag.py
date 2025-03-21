import asyncio
from lightrag import QueryParam
from build_graph import initialize_rag


if __name__ == "__main__":

    rag = asyncio.run(initialize_rag())

    output = rag.query(
        "What is SQuAD?",
        param=QueryParam(mode="mix")
    )
    print(output)

    output = rag.query(
        "What is the default CPU configuration for a created endpoint?",
        param=QueryParam(mode="mix")
    )
    print(output)