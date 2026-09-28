
from rag import get_embedded_chunks, retrieve_chunks


# -------------------------
# 1. Evaluation Dataset
# -------------------------

# We will add manually reviewed relevant chunk IDs here.
# Leave them empty until we inspect the retrieved passages.

TEST_CASES = [
    {
        "question": "Who is Odysseus's son?",
        "relevant_chunk_ids": [],
    },
    {
        "question": "Who is Penelope?",
        "relevant_chunk_ids": [],
    },
    {
        "question": "Why did Odysseus leave Ithaca?",
        "relevant_chunk_ids": [],
    },
    {
        "question": "Why were the suitors staying in Odysseus's house?",
        "relevant_chunk_ids": [],
    },
    {
        "question": "What happened to Odysseus's crew?",
        "relevant_chunk_ids": [],
    },
    {
        "question": "Who is Telemachus?",
        "relevant_chunk_ids": [],
    },
    {
        "question": "Who is Athena?",
        "relevant_chunk_ids": [],
    },
    {
        "question": "Who is Circe?",
        "relevant_chunk_ids": [],
    },
]


# -------------------------
# 2. Inspect Retrieved Chunks
# -------------------------

def inspect_retrieval(embedded_chunks, top_k=5):
    print("\n" + "=" * 60)
    print("RETRIEVAL INSPECTION")
    print("=" * 60)

    for test_case in TEST_CASES:
        question = test_case["question"]

        print(f"\nQuestion: {question}")

        results = retrieve_chunks(
            question,
            embedded_chunks,
            top_k=top_k
        )

        for rank, result in enumerate(results, start=1):
            print(
                f"\nRank: {rank}"
                f" | Chunk ID: {result['id']}"
                f" | Score: {result['score']:.4f}"
            )

            print(
                result["text"][:500].replace("\n", " ")
            )

        print("\n" + "-" * 60)


# -------------------------
# 3. Calculate Metrics
# -------------------------

def evaluate_retrieval(embedded_chunks, top_k=5):
    recall_scores = []
    reciprocal_ranks = []

    print("\n" + "=" * 60)
    print("RETRIEVAL BENCHMARK")
    print("=" * 60)

    for test_case in TEST_CASES:
        question = test_case["question"]
        relevant_ids = set(test_case["relevant_chunk_ids"])

        if not relevant_ids:
            print(
                f"\nSkipping: {question}\n"
                "No relevant chunk IDs have been labelled."
            )
            continue

        results = retrieve_chunks(
            question,
            embedded_chunks,
            top_k=top_k
        )

        retrieved_ids = [item["id"] for item in results]

        relevant_retrieved = set(retrieved_ids) & relevant_ids

        # Recall@K = relevant retrieved chunks / all labelled
        # relevant chunks for this question.
        recall = len(relevant_retrieved) / len(relevant_ids)

        # Reciprocal rank of the first relevant retrieved chunk.
        reciprocal_rank = 0.0

        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in relevant_ids:
                reciprocal_rank = 1.0 / rank
                break

        recall_scores.append(recall)
        reciprocal_ranks.append(reciprocal_rank)

        print(f"\nQuestion: {question}")
        print(f"Relevant IDs: {sorted(relevant_ids)}")
        print(f"Retrieved IDs: {retrieved_ids}")
        print(f"Recall@{top_k}: {recall:.3f}")
        print(f"Reciprocal rank: {reciprocal_rank:.3f}")

    if not recall_scores:
        print(
            "\nNo metrics calculated. "
            "Inspect passages and label relevant chunk IDs first."
        )
        return

    mean_recall = sum(recall_scores) / len(recall_scores)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Evaluated questions: {len(recall_scores)}")
    print(f"Mean Recall@{top_k}: {mean_recall:.3f}")
    print(f"MRR@{top_k}: {mrr:.3f}")


# -------------------------
# 4. Run Evaluation
# -------------------------

if __name__ == "__main__":
    embedded_chunks = get_embedded_chunks()

    # First, inspect the retrieved passages.
    # After labelling relevant IDs, you can run the metrics.
    inspect_retrieval(embedded_chunks, top_k=5)

    evaluate_retrieval(embedded_chunks, top_k=5)