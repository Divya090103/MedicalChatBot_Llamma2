from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from pinecone import Pinecone, ServerlessSpec
# from langchain.llms import CTransformers
from langchain_community.llms import CTransformers
# from azure.functions import ServerlessSpec
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


# initialising flask
app = Flask(__name__)
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

pc = Pinecone(
    api_key=PINECONE_API_KEY,  # Use the API key retrieved from environment variables
    serverless_spec=ServerlessSpec(
        cloud='aws',
        region='us-west-2'  # Specify the region you're working with
    )
)


# Now you can interact with the Pinecone index using the instance `pc`
index_name = "mchatbot"

# Check if the index exists; if not, create it
# if index_name not in pc.list_indexes():
#     pc.create_index(index_name, dimension=384)  # Specify the correct dimension


index = pc.Index(index_name)
# index = pc[index_name]


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}
# LOAD THE LLAMA MODEL
llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens':512,
                            'temperature':0.8
                            }
                    )


# Download embeddings model from HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# retriever = Pinecone(index, embeddings.embed_query, {"text": "text"})  # Specify metadata key used for the text
# Initialize the Pinecone VectorStore correctly
retriever = Pinecone(
    index=index,
    embedding_function=embeddings.embed_query,  # Ensure embed_query is the correct function
    text_key="text"  # Metadata key for the text
)

# Creating the RetrievalQA object
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever.as_retriever(search_kwargs={'k': 2}),  # Define number of documents to retrieve
#     return_source_documents=True,
#     chain_type_kwargs=chain_type_kwargs
# )

# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",  # Specify the correct chain type
#     chain_type_kwargs={
#         "retriever": retriever,  # Pass the configured retriever as a keyword argument
#         "prompt": prompt_template  # Pass the prompt template as a keyword argument
#     },
#     return_source_documents=True,
# )


# Creating the RetrievalQA object
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Specify the correct chain type
    retriever=retriever.as_retriever(search_kwargs={'k': 2}),  # Correct way to pass the retriever
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": prompt_template,  # Ensure the prompt is passed correctly
    }
)


# creating default route for flask app
@app.route("/")
def index():
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug = True)