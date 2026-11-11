import pytest
from src.evaluation.evaluator import RAGEvaluator


def test_rag_evaluator_basic():
    evaluator = RAGEvaluator()

    query = "What is the capital of France?"
    contexts = [
        "France is a country in Western Europe. Its capital and largest city is Paris.",
        "Berlin is the capital of Germany located in central Europe."
    ]
    answer = "The capital of France is Paris, located in Western Europe."
    sources = [{"id": "doc1", "title": "European Capitals Guide"}]

    result = evaluator.evaluate_sample(
        query=query,
        retrieved_contexts=contexts,
        answer=answer,
        sources=sources
    )

    assert "retrieval_relevance" in result
    assert "answer_faithfulness" in result
    assert "context_relevance" in result
    assert "source_attribution" in result
    assert "aggregate_score" in result
    assert result["aggregate_score"] > 0.5


def test_batch_evaluation():
    evaluator = RAGEvaluator()
    samples = [
        {
            "query": "Explain quantum entanglement.",
            "retrieved_contexts": ["Quantum entanglement is a physical phenomenon where particles interact."],
            "answer": "Quantum entanglement describes particles interacting across distance.",
            "sources": [{"title": "Physics Primer"}]
        }
    ]

    summary = evaluator.evaluate_batch(samples)
    assert summary["total_samples"] == 1
    assert "average_aggregate_score" in summary