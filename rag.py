
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


# -------------------------
# 1. Document Loading
# -------------------------

def load_document(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def clean_document(text):
    start_marker = (
        "*** START OF THE PROJECT GUTENBERG EBOOK THE ODYSSEY ***"
    )
    end_marker = (
        "*** END OF THE PROJECT GUTENBERG EBOOK THE ODYSSEY ***"
    )

    start = text.find(start_marker)
    end = text.find(end_marker)

    if start != -1:
        text = text[start + len(start_marker):]

    if end != -1:
        text = text[:end]

    return text.strip()


# -------------------------
# 2. Text Chunking
# -------------------------

def chunk_text(text, chunk_size=1000, overlap=200):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size."
        )

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


# -------------------------
# 3. Embedding Generation
# -------------------------

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

        print(f"Embedding chunk {i + 1}/{len(chunks)}")

        embedding = create_embedding(chunk)

        embedded_chunks.append({
            "id": i,
            "text": chunk,
            "embedding": embedding
        })

        save_embeddings(embedded_chunks, file_path)

        print(
            f"Saved progress: "
            f"{len(embedded_chunks)} / {len(chunks)}"
        )

    return embedded_chunks


def save_embeddings(data, file_path):
    directory = os.path.dirname(file_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def load_embeddings(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    for i, item in enumerate(data):
        if "id" not in item:
            item["id"] = i

    return data


def get_embedded_chunks():
    text = load_document("data/the-odyssey.txt")
    text = clean_document(text)
    chunks = chunk_text(text)

    embedding_file = "data/embeddings.pkl"

    if os.path.exists(embedding_file):
        embedded_chunks = load_embeddings(embedding_file)

        if len(embedded_chunks) == len(chunks):
            print("Loading saved embeddings...")
            print(f"Loaded {len(embedded_chunks)} embeddings.")
            return embedded_chunks

        print("Embedding file is incomplete.")
    else:
        print("No embedding cache found.")

    embedded_chunks = create_chunk_embeddings(
        chunks,
        embedding_file
    )

    if len(embedded_chunks) != len(chunks):
        raise ValueError(
            f"Expected {len(chunks)} embeddings, "
            f"but found {len(embedded_chunks)}."
        )

    return embedded_chunks


# -------------------------
# 4. Similarity & Retrieval
# -------------------------

def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def retrieve_chunks(question, embedded_chunks, top_k=5):
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    question_embedding = create_embedding(question)

    results = []

    for item in embedded_chunks:
        score = cosine_similarity(
            question_embedding,
            item["embedding"]
        )

        results.append({
            "id": item["id"],
            "text": item["text"],
            "score": score
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:top_k]


# -------------------------
# 5. Context Formatting
# -------------------------

def format_context(retrieved_chunks):
    context_parts = []

    for item in retrieved_chunks:
        context_parts.append(
            f"""
SOURCE CHUNK {item['id']}
SIMILARITY SCORE: {item['score']:.4f}

{item['text']}
"""
        )

    return "\n".join(context_parts)


# -------------------------
# 6. Answer Generation
# -------------------------

def generate_answer(question, retrieved_chunks):
    context = format_context(retrieved_chunks)

    prompt = f"""
You are a question-answering assistant for Homer's The Odyssey.

Answer the user's question using ONLY the retrieved context below.

Rules:
1. Use only information contained in the provided context.
2. Do not use outside knowledge.
3. Do not invent or assume unsupported facts.
4. If the context does not contain enough information, say:
   "I could not find the answer in the provided text."
5. Give a concise answer.
6. When possible, mention the relevant source chunk IDs.

Retrieved context:

{context}

Question:

{question}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text


# -------------------------
# 7. Retrieval Evaluation
# -------------------------

def evaluate_retrieval(embedded_chunks):
    test_questions = [
        "Who is Odysseus's son?",
        "Who is Penelope?",
        "Why did Odysseus leave Ithaca?",
        "Why were the suitors staying in Odysseus's house?",
        "What happened to Odysseus's crew?",
        "Who is Telemachus?",
        "Who is Athena?",
        "Who is Circe?",
    ]

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)

    for question in test_questions:
        print(f"\nQuestion: {question}")

        results = retrieve_chunks(
            question,
            embedded_chunks,
            top_k=5
        )

        for rank, result in enumerate(results, start=1):
            print(
                f"\nRank: {rank}"
                f" | Chunk ID: {result['id']}"
                f" | Score: {result['score']:.4f}"
            )

            print(
                result["text"][:300].replace("\n", " ")
            )

        print("\n" + "-" * 60)


# -------------------------
# 8. Interactive Application
# -------------------------

def main():
    text = load_document("data/the-odyssey.txt")
    text = clean_document(text)

    print("Characters:", len(text))

    chunks = chunk_text(text)
    print("Number of chunks:", len(chunks))

    embedded_chunks = get_embedded_chunks()

    print("Number of embedded chunks:", len(embedded_chunks))

    while True:
        question = input(
            "\nAsk about The Odyssey (or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        results = retrieve_chunks(
            question,
            embedded_chunks,
            top_k=5
        )

        print("\nRetrieved Context:")

        for rank, result in enumerate(results, start=1):
            print(f"\n--- Rank {rank} ---")
            print(f"Chunk ID: {result['id']}")
            print(f"Similarity: {result['score']:.4f}")
            print(result["text"][:500])

        answer = generate_answer(question, results)

        print("\nAnswer")
        print(answer)


if __name__ == "__main__":
    main()