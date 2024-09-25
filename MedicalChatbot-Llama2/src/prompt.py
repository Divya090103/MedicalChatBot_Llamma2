# prompt_template = """
# Use the following pieces of information to answer the user's question.
# If you don't find the answer just say you don't know, don't try to make up an answer.

# Context: {context}
# Question: {question}

# Only return the helpful answer below and nothing else.
# Helpful answer:
# """


prompt_template = """
Use the following pieces of information and the chat history to answer the user's question.
If you don't know the answer, simply state that you don't know; avoid making up an answer.
Keep the answer short and crisp

Question: {question}
Information: {context}
History: {history}

If there is nothing in the history, Please ignore it.


Helpful Answer:
"""