from google import genai
from dotenv import load_dotenv
import os

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

embedding = create_embedding(
    "Odysseus returns to Ithaca after many years."
)

print('Embedding dimensions:', len(embedding))
print(embedding[:10])

text = load_document("data/the-odyssey.txt")

print('Characters:', len(text))

chunks = chunk_text(text)

print("Number of chunks:", len(chunks))
print("\nFirst Chunk:\n")
print(chunks[0])