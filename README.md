# Odyssey RAG

### A Retrieval-Augmented Generation (RAG) System for Homer's *The Odyssey*

Odyssey RAG is a Python-based Retrieval-Augmented Generation application that lets users ask questions about *The Odyssey* and receive answers grounded in the source text.

The project uses Google's Gemini models to generate embeddings and answer questions, with NumPy-based cosine similarity for semantic retrieval. It also includes an initial evaluation framework to inspect retrieved passages and measure retrieval quality.

The goal is to build a transparent, grounded question-answering system while learning the fundamentals of LLM application engineering.

---

## Features

* **Document processing:** Loads and cleans the Project Gutenberg text of *The Odyssey*.
* **Text chunking:** Splits the document into overlapping chunks to preserve contextual continuity.
* **Semantic embeddings:** Uses Google's Gemini Embedding 2 model to represent text as numerical vectors.
* **Embedding persistence:** Stores chunk embeddings locally using pickle to avoid regenerating them on every run.
* **Resumable embedding generation:** Saves progress during embedding generation so interrupted runs can resume.
* **Semantic search:** Retrieves relevant passages using cosine similarity.
* **Context formatting:** Organizes retrieved passages with source chunk IDs and similarity scores.
* **Grounded answer generation:** Uses Gemini 3.5 Flash to answer questions based on retrieved passages.
* **Retrieval evaluation:** Includes a test-question dataset, retrieval inspection, Recall@K, and Mean Reciprocal Rank (MRR).

## Tech Stack

| Technology         | Purpose                                 |
| ------------------ | --------------------------------------- |
| Python             | Application logic                       |
| Google Gen AI SDK  | Gemini API integration                  |
| Gemini Embedding 2 | Text and query embeddings               |
| Gemini 3.5 Flash   | Answer generation                       |
| NumPy              | Cosine similarity and vector operations |
| Pickle             | Local embedding cache                   |
| python-dotenv      | Environment variable management         |
| Git & GitHub       | Version control                         |

## How It Works

```text
                 DOCUMENT INGESTION

  The Odyssey (text file)
           |
           v
    Clean the document
           |
           v
    Split into chunks
    (1,000 characters)
    (200-character overlap)
           |
           v
    Generate embeddings
    (Gemini Embedding 2)
           |
           v
    Save embeddings locally
    (data/embeddings.pkl)


                 QUESTION ANSWERING

       User question
           |
           v
    Embed the question
           |
           v
    Compare with document
    embeddings using cosine
    similarity
           |
           v
    Retrieve top 5 chunks
           |
           v
    Format retrieved context
    with source chunk IDs
           |
           v
    Gemini 3.5 Flash
           |
           v
    Grounded answer
```

## Project Structure

```text
odyssey-rag/
├── data/
│   ├── the-odyssey.txt
│   └── embeddings.pkl
├── rag.py
├── evaluation.py
├── .env
├── .gitignore
└── README.md
```

**Important:** `data/embeddings.pkl` is a generated cache and should remain excluded from Git. The `.env` file must also be excluded because it contains the Gemini API key.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/vaibhav-barman/odyssey-rag.git
cd odyssey-rag
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install google-genai python-dotenv numpy
```

### 4. Configure your API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Obtain an API key through [Google AI Studio](https://aistudio.google.com/).

Never commit your API key or share it publicly.

### 5. Prepare the source document

Place the plain-text edition of *The Odyssey* in:

```text
data/the-odyssey.txt
```

The application cleans the Project Gutenberg header and footer before chunking the text.

### 6. Run the application

```bash
python rag.py
```

On the first run, the application generates and caches the document embeddings. Subsequent runs reuse the cache when it is available.

You can then ask questions about the text through the interactive terminal interface.

## Retrieval Configuration

The current document-processing configuration is:

| Parameter                |                Value |
| ------------------------ | -------------------: |
| Cleaned document length  |   679,669 characters |
| Chunk size               |     1,000 characters |
| Chunk overlap            |       200 characters |
| Generated chunks         |                  850 |
| Embedding model          | `gemini-embedding-2` |
| Embedding dimensionality |                  768 |
| Default retrieval count  |             5 chunks |
| Answer generation model  |   `gemini-3.5-flash` |

These values describe the current project configuration and can be changed as the system evolves.

## Retrieval Evaluation

The project includes `evaluation.py`, which separates retrieval inspection and evaluation from the normal chatbot workflow.

### Inspect retrieved passages

Run:

```bash
python evaluation.py
```

The script evaluates eight initial questions, including:

* Who is Odysseus's son?
* Who is Penelope?
* Why did Odysseus leave Ithaca?
* Why were the suitors staying in Odysseus's house?
* What happened to Odysseus's crew?
* Who is Telemachus?
* Who is Athena?
* Who is Circe?

For each question, the script displays the top five retrieved passages, their chunk IDs, and their similarity scores.

### Evaluation metrics

**Recall@K**

Measures the proportion of manually labelled relevant chunks that appear among the top K retrieved results.

**Mean Reciprocal Rank (MRR@K)**

Measures how highly the first relevant chunk appears in the retrieved results, averaged across evaluated questions.

The current evaluation dataset has empty `relevant_chunk_ids` lists. These must be manually labelled after inspecting the retrieved passages. Until then, the script intentionally skips metric calculations.

The current evaluation framework measures retrieval quality; it does not yet provide a comprehensive evaluation of answer correctness or faithfulness.

## Current Limitations

* Retrieval uses basic cosine similarity rather than a dedicated vector database.
* Embeddings are stored locally in a pickle file.
* Retrieval evaluation requires manually labelled relevant chunk IDs.
* The evaluation dataset is small and intended for initial experimentation.
* The application runs through a terminal interface.
* The project does not yet include a deployed API, web interface, automated evaluation pipeline, or production monitoring.

## Roadmap

The following are planned improvements, not completed features.

* [ ] Label relevant chunks for the evaluation dataset.
* [ ] Establish a baseline using Recall@5 and MRR@5.
* [ ] Improve retrieval through experiments with chunk sizes, overlap, and retrieval parameters.
* [ ] Evaluate generated answers for correctness and faithfulness.
* [ ] Explore a vector database such as Chroma or PostgreSQL with pgvector.
* [ ] Add a FastAPI backend.
* [ ] Add automated tests and structured logging.
* [ ] Containerize the application with Docker.
* [ ] Deploy the application and document its operational requirements.

## Learning Objectives

This project is an ongoing exploration of practical AI engineering concepts:

* Document ingestion and preprocessing
* Embedding-based semantic search
* Retrieval-Augmented Generation
* Context construction and source attribution
* Grounding LLM responses in retrieved evidence
* Retrieval evaluation and benchmarking
* API integration, caching, and error handling
* Building toward testable and deployable AI applications

## Author

**Vaibhav Barman**

B.Sc. Computer Science student at BITS Pilani, focusing on AI engineering, LLM applications, and Generative AI.

* **GitHub:** [vaibhav-barman](https://github.com/vaibhav-barman)
* **Project repository:** [odyssey-rag](https://github.com/vaibhav-barman/odyssey-rag)

---

*Odyssey RAG is an educational project exploring retrieval-augmented generation. It is a work in progress, with retrieval evaluation and production-readiness improvements still underway.*
