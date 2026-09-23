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

text = load_document("data/the-odyssey.txt")

print('Characters:', len(text))

chunks = chunk_text(text)

print("Number of chunks:", len(chunks))
print("\nFirst Chunk:\n")
print(chunks[0])