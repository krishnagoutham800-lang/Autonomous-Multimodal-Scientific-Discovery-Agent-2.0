"""
Comprehensive Unit Tests for the RAG Evaluation Module.
Covers:
1. Standard successful evaluation sample
2. Contradictory claims detection
3. Unsupported claims detection
4. Valid vs. Invalid source references (Pipeline format: (Source X) + paper IDs + page metadata)
5. Empty contexts and answers (Input validation)
6. Missing source metadata
7. Isolated metric execution failure recovery
8. Batch evaluation and faithfulness gating
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path even when run directly as a script
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.metrics import (
    RetrievalRelevanceMetric,
    AnswerFaithfulnessMetric,
    ContextRelevanceMetric,
    SourceAttributionMetric,
)


class FaultyMockMetric:
    """Mock metric designed to raise an exception to test isolated metric failure recovery."""
    def evaluate(self, query, retrieved_contexts, answer, sources=None, ground_truth=None):
        raise RuntimeError("Simulated metric failure")


def test_standard_evaluation_flow():
    evaluator = RAGEvaluator()

    query = "What is quantum entanglement?"
    contexts = [
        "Quantum entanglement is a physical phenomenon where pairs of particles interact in ways such that the quantum state of each particle cannot be described independently."
    ]
    answer = "Quantum entanglement describes linked particles whose states cannot be described independently (Source 1)."
    sources = [{"id": "doc1", "paper_id": "P12345", "page": 12, "title": "Quantum Mechanics"}]

    result = evaluator.evaluate_sample(
        query=query,
        retrieved_contexts=contexts,
        answer=answer,
        sources=sources
    )

    assert "scores" in result
    assert "aggregate_score" in result
    assert result["scores"]["retrieval_relevance"] > 0.0
    assert result["scores"]["answer_faithfulness"] > 0.0
    assert result["scores"]["source_attribution"] == 1.0


def test_contradictory_claims_detection():
    metric = AnswerFaithfulnessMetric()
    query = "Is the compound stable?"
    contexts = [
        "Laboratory tests confirm that the compound is extremely unstable and decomposes rapidly at room temperature."
    ]
    # The answer makes a contradictory claim while sharing vocabulary
    answer = "The compound is stable and does not decompose at room temperature."

    res = metric.evaluate(query=query, retrieved_contexts=contexts, answer=answer)
    # Faithfulness score should be penalized due to contradiction / negation mismatch
    assert res["score"] < 1.0


def test_unsupported_claims_detection():
    metric = AnswerFaithfulnessMetric()
    query = "What is the capital of France?"
    contexts = [
        "France is a country in Western Europe. Its official language is French."
    ]
    # Answer hallucinates completely ungrounded facts not present in context
    answer = "The capital of France is Paris, and its currency is the Euro, established in 1999."

    res = metric.evaluate(query=query, retrieved_contexts=contexts, answer=answer)
    assert res["score"] < 1.0


def test_source_references_valid_and_invalid():
    metric = SourceAttributionMetric()
    contexts = ["Some context."]
    
    sources = [
        {"id": "doc1", "paper_id": "PAPER-001", "page": 5},
        {"id": "doc2", "paper_id": "PAPER-002", "page": 10}
    ]

    # Valid citation matching pipeline format (Source 1)
    valid_answer = "According to recent findings (Source 1), the results are conclusive."
    res_valid = metric.evaluate(query="test", retrieved_contexts=contexts, answer=valid_answer, sources=sources)
    assert res_valid["score"] == 0.5  # 1 out of 2 sources referenced correctly

    # Invalid citation referencing non-existent sources
    invalid_answer = "Based on external data (Source 99), everything is fine."
    res_invalid = metric.evaluate(query="test", retrieved_contexts=contexts, answer=invalid_answer, sources=sources)
    assert res_invalid["score"] == 0.0


def test_empty_contexts_and_answers_validation():
    evaluator = RAGEvaluator()

    # Empty query should raise ValueError
    with pytest.raises(ValueError, match="query"):
        evaluator.evaluate_sample(
            query="",
            retrieved_contexts=["Valid context"],
            answer="Valid answer"
        )

    # Empty answer should raise ValueError
    with pytest.raises(ValueError, match="answer"):
        evaluator.evaluate_sample(
            query="Valid query",
            retrieved_contexts=["Valid context"],
            answer="   "
        )

    # Invalid retrieved_contexts type should raise TypeError
    with pytest.raises(TypeError, match="retrieved_contexts"):
        evaluator.evaluate_sample(
            query="Valid query",
            retrieved_contexts="not-a-list",
            answer="Valid answer"
        )


def test_missing_metadata():
    evaluator = RAGEvaluator()
    query = "What is gravity?"
    contexts = ["Gravity is a fundamental force."]
    answer = "Gravity pulls objects together."
    
    # Sources omitted (None) should pass cleanly without failing attribution
    result = evaluator.evaluate_sample(
        query=query,
        retrieved_contexts=contexts,
        answer=answer,
        sources=None
    )
    assert result["scores"]["source_attribution"] == 1.0


def test_isolated_metric_failures():
    evaluator = RAGEvaluator()
    # Inject a faulty mock metric into the evaluator instance
    evaluator.metrics["faulty_metric"] = FaultyMockMetric()

    query = "Test query"
    contexts = ["Test context"]
    answer = "Test answer"

    result = evaluator.evaluate_sample(
        query=query,
        retrieved_contexts=contexts,
        answer=answer
    )

    # Verify that the failed metric is caught and recorded without crashing other metrics
    assert result["scores"]["faulty_metric"] is None
    assert "faulty_metric" in result["failed_metrics"]
    assert result["scores"]["retrieval_relevance"] is not None
    assert result["scores"]["answer_faithfulness"] is not None


def test_batch_evaluation_and_faithfulness_gating():
    evaluator = RAGEvaluator(faithfulness_threshold=0.5)

    batch_samples = [
        {
            "query": "What is AI?",
            "retrieved_contexts": ["Artificial Intelligence is machine simulation."],
            "answer": "AI simulates human intelligence (Source 1).",
            "sources": [{"id": "1", "paper_id": "AI-101"}]
        },
        {
            "query": "What is Python?",
            "retrieved_contexts": ["Python is a programming language."],
            "answer": "Python is a snake found in tropical rainforests.", # Hallucination / unfaithful
            "sources": [{"id": "1", "paper_id": "BIO-202"}]
        }
    ]

    summary = evaluator.evaluate_batch(batch_samples)
    assert summary["total_samples"] == 2
    assert "average_aggregate_score" in summary
    assert summary["faithfulness_flagged_samples"] >= 1