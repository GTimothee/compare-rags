evaluation_prompt = """Your task is to evaluate a Q/A system. 
The user will give you a question, an expected answer and the system's response.
You will evaluate the system's response and provide a score and a text explanation for the score you gave.
We are asking ourselves if the response is correct, accurate and factual, based on the reference answer.

Guidelines:
1. Write a detailed feedback that assess the quality of the response strictly based on the given scores description, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the scores description.
3. Follow the JSON format provided below for your output.

Scores description:
    Score 1: The response is completely incorrect, inaccurate, and/or not factual.
    Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
    Score 3: The response is somewhat correct, accurate, and/or factual.
    Score 4: The response is mostly correct, accurate, and factual.
    Score 5: The response is completely correct, accurate, and factual.

Output Format (JSON only):
{{
    "evaluation": {{
        "feedback": "(your rationale for the rating, as a text)",
        "score": (your rating, as a number between 1 and 5)
    }}
}}

Do not include any additional text—only the JSON object. Any extra content will result in a grade of 0."""