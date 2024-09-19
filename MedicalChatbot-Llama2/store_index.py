from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from dotenv import load_dotenv
# from langchain.vectorstores import Pinecone
from langchain_community.vectorstores import Pinecone
import pinecone
import os
from pinecone import Pinecone, ServerlessSpec


load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# print(PINECONE_API_KEY)

extracted_data = load_pdf("data/")
print(extracted_data)



text_chunks = text_split(extracted_data)
print("Length of chunks: ", len(text_chunks))
print(text_chunks)

embeddings = download_hugging_face_embeddings()


# Correctly pass the string name of the environment variable to `os.getenv`
# api_key = os.getenv("PINECONE_API_KEY")  # Retrieve the API key from environment variable
api_key=PINECONE_API_KEY
# Check if the API key was retrieved successfully
if not api_key:
    raise ValueError("PINECONE_API_KEY environment variable not set!")

# Create Pinecone instance
pc = Pinecone(
    api_key=api_key,  # Use the API key retrieved from environment variables
    serverless_spec=ServerlessSpec(
        cloud='aws',
        region='us-west-2'  # Specify the region you're working with
    )
)


# Now you can interact with the Pinecone index using the instance `pc`
index_name = "mchatbot"
# Create the index or connect to an existing one
index = pc.Index(index_name)
index

# BELOW COMMENTED CODE IS FOR THE SITUATION WHEN WHOLE 7000 CHUNKS ARE NOT UPLOADED IN A SINGLE GO

# START_INDEX = 6452
# def store_embeddings_in_pinecone(text_chunks, embeddings):
#     total_chunks = len(text_chunks)
#     for i in range(START_INDEX, total_chunks):
#         chunk = text_chunks[i]
#         text = chunk.page_content  # Extract the actual text content of the chunk
#         embedding = embeddings.embed_documents([text])[0]  # Generate the embedding for this chunk
#         embedding_id = f"chunk_{i}"  # Create a unique ID for each chunk
        
#         # Upsert the embedding into the Pinecone index along with metadata
#         index.upsert(vectors=[(embedding_id, embedding, {"text": text})])  # Store text in metadata
#         print(f"Upserted chunk {i} with ID: {embedding_id} and metadata")


# Create embeddings and store each text chunk into Pinecone along with metadata
# def store_embeddings_in_pinecone(text_chunks, embeddings):
#     for i, chunk in enumerate(text_chunks):
#         text = chunk.page_content  # Extract the actual text content of the chunk
#         embedding = embeddings.embed_documents([text])[0]  # Generate the embedding for this chunk
#         embedding_id = f"chunk_{i}"  # Create a unique ID for each chunk

#         # Upsert the embedding into the Pinecone index along with metadata
#         index.upsert(vectors=[(embedding_id, embedding, {"text": text})])  # Store text in metadata
#         print(f"Upserted chunk {i} with ID: {embedding_id} and metadata")



# Download embeddings model from HuggingFace
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

