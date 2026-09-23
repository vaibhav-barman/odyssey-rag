from google import genai
from dotenv import load_dotenv
import os
import pickle

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    return text

def chunk_text(text, chunk_size=1000):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i: i + chunk_size]
        chunks.append(chunk)

    return chunks

def create_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return result.embeddings[0].values

def create_chunk_embeddings(chunks):
    embedded_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"Embedded chunk {i+1}/{len(chunks)}")

        embedding = create_embedding(chunk)

        embedded_chunks.append(
            {
                "text" : chunk, 
                "embedding" : embedding
            }
        )

    return embedded_chunks

def save_embeddings(data, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(data, file)

def load_embeddings(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

text = load_document("data/the-odyssey.txt")

print('Characters:', len(text))

chunks = chunk_text(text)

print("Number of chunks:", len(chunks))
# print("\nFirst Chunk:\n")
# print(chunks[0])

embedding_file = "data/embeddings.pkl"

if os.path.exists(embedding_file):
    print("Loading saved embeddings...")
    embedded_chunks = load_embeddings(embedding_file)

else:
    print("Creating embeddings...")
    embedded_chunks = create_chunk_embeddings(chunks)

    save_embeddings(
        embedded_chunks,
        embedding_file
    )

print("Number of embedded chunks:", len(embedded_chunks))