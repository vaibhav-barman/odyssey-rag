def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    return text

text = load_document("data/the-odyssey.txt")

print('Characters:', len(text))
print(text[:2000])