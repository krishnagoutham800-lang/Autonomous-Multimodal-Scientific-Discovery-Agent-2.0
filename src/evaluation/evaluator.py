"""
RAG Evaluator Orchestrator
"""

from typing import Any, Dict, List, Optional

# Changed from relative import to absolute package import
from src.evaluation.metrics import (
    AnswerFaithfulnessMetric,
    ContextRelevanceMetric,
    RetrievalRelevanceMetric,
    SourceAttributionMetric,
)


class RAGEvaluator:
    """Orchestrates running evaluation suites on RAG pipeline outputs."""

    def __init__(self):
        self.metrics = {
            "retrieval_relevance": RetrievalRelevanceMetric(),
            "answer_faithfulness": AnswerFaithfulnessMetric(),
            "context_relevance": ContextRelevanceMetric(),
            "source_attribution": SourceAttributionMetric(),
        }

    def evaluate_sample(
        self,
        query: str,
        retrieved_contexts: List[str],
        answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluates a single RAG generation sample across all core metrics."""
        results = {}
        total_score = 0.0

        for name, metric in self.metrics.items():
            res = metric.evaluate(
                query=query,
                retrieved_contexts=retrieved_contexts,
                answer=answer,
                sources=sources,
                ground_truth=ground_truth,
            )
            results[name] = res
            total_score += res.get("score", 0.0)

        results["aggregate_score"] = round(total_score / len(self.metrics), 4)
        return results

    def evaluate_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluates a batch of RAG samples and computes macro averages."""
        if not samples:
            return {"average_aggregate_score": 0.0, "total_samples": 0}

        batch_results = []
        metric_sums = {name: 0.0 for name in self.metrics.keys()}
        aggregate_sum = 0.0

        for sample in samples:
            res = self.evaluate_sample(
                query=sample.get("query", ""),
                retrieved_contexts=sample.get("retrieved_contexts", []),
                answer=sample.get("answer", ""),
                sources=sample.get("sources"),
                ground_truth=sample.get("ground_truth"),
            )
            batch_results.append(res)
            aggregate_sum += res["aggregate_score"]
            for name in self.metrics.keys():
                metric_sums[name] += res[name]["score"]

        n = len(samples)
        summary = {
            "total_samples": n,
            "average_aggregate_score": round(aggregate_sum / n, 4),
            "metric_averages": {
                name: round(metric_sums[name] / n, 4) for name in self.metrics.keys()
            },
            "detailed_results": batch_results,
        }
        return summary