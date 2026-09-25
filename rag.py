from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import pickle
import numpy as np

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    return text

def clean_document(text):
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK THE ODYSSEY ***"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK THE ODYSSEY ***"

    start = text.find(start_marker)
    end = text.find(end_marker)

    if start != -1:
        text = text[start + len(start_marker):]

    if end != -1:
        text = text[:end]

    return text.strip()

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def create_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=768
        )
    )

    return result.embeddings[0].values

def create_chunk_embeddings(chunks, file_path):
    embedded_chunks = []

    if os.path.exists(file_path):
        print("Found existing embedding progress.")
        embedded_chunks = load_embeddings(file_path)

        print(
            f"Already embedded: "
            f"{len(embedded_chunks)} / {len(chunks)}"
        )

    start_index = len(embedded_chunks)

    for i in range(start_index, len(chunks)):
        chunk = chunks[i]
        print(
            f"Embedding chunk "
            f"{i+1}/{len(chunks)}"
        )

        embedding = create_embedding(chunk)

        embedded_chunks.append(
            {
                "id": i,
                "text": chunk,
                "embedding": embedding
            }
        )

        save_embeddings(
            embedded_chunks,
            file_path
        )

        print(
            f"Saved Progress: "
            f"{len(embedded_chunks)} / {len(chunks)}"
        )

    return embedded_chunks

def save_embeddings(data, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(data, file)

def load_embeddings(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    for i, item in enumerate(data):
        if "id" not in item:
            item["id"] = i

    return data

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

def retrieve_chunks(question, embedded_chunks, top_k=5):
    question_embedding = create_embedding(question)

    results = []

    for item in embedded_chunks:
        score = cosine_similarity(
            question_embedding, 
            item["embedding"]
        )

        results.append(
            {
                "id": item["id"],
                "text": item["text"],
                "score": score
            }
        )

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]

def generate_answer(question, retrieved_chunks):
    context = "\n\n".join(
        item["text"]
        for item in retrieved_chunks
    )

    prompt = f"""
        You are a question-answering assistant for Homer's The Odyssey.

        Answer the user's question using ONLY the provided context.

        If the answer cannot be found in the context, say:
        "I could not find the answer in the provided text."

        Do not use outside knowledge.
        Do not invent information.

        Context: {context}

        Question: {question}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text

text = load_document("data/the-odyssey.txt")

text = clean_document(text)

print('Characters:', len(text))

chunks = chunk_text(text)

print("Number of chunks:", len(chunks))
# print("\nFirst Chunk:\n")
# print(chunks[0])

embedding_file = "data/embeddings.pkl"

if os.path.exists(embedding_file):
    embedded_chunks = load_embeddings(embedding_file)

    if len(embedded_chunks) == len(chunks):
        print("Loading saved embedding ... ")
        print(f"Loaded {len(embedded_chunks)} embeddings.")

    else:
        print("Embedding file is incomplete")
        embedded_chunks = create_chunk_embeddings(
            chunks,
            embedding_file
        )
else:
    print("Creating embeddings...")
    embedded_chunks = create_chunk_embeddings(
        chunks,
        embedding_file
    )

print("Number of embedded chunks:", len(embedded_chunks))

while True:
    question = input("\nAsk about The Odyssey (or type 'exit'): ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    results = retrieve_chunks(
        question, 
        embedded_chunks, 
        top_k=5
    )

    print("\nRetrieved Context: ")

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Rank {rank} ---")
        print(f"Chunk ID: {result['id']}")
        print(f"Similarity: {result['score']:.4f}")
        print(result["text"][:500])

    answer = generate_answer(
        question, 
        results
    )

    print("\nAnswer")
    print(answer)