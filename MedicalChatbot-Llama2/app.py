from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
# from azure.functions import ServerlessSpec


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

index = pc.Index(index_name)


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}
# LOAD THE LLAMA MODEL
llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens':512,
                            'temperature':0.8
                            }
                    )

retriever = Pinecone(index, embeddings.embed_query, {"text": "text"})  # Specify metadata key used for the text

# Creating the RetrievalQA object
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever.as_retriever(search_kwargs={'k': 2}),  # Define number of documents to retrieve
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)


# creating default route for flask app
@app.route("/")
def index():
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug = True)