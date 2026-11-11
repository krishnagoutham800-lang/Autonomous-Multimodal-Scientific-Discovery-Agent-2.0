import unittest

from src.evaluation import (
    context_relevance,
    evaluate_rag,
    faithfulness,
    retrieval_metrics,
    source_attribution,
)


class RetrievalMetricsTests(unittest.TestCase):
    def test_ranking_metrics_and_graded_ndcg(self):
        scores = retrieval_metrics({"a": 2, "b": 1}, ["x", "a", "b"], k=2)
        self.assertEqual(scores["precision"], 0.5)
        self.assertEqual(scores["recall"], 0.5)
        self.assertEqual(scores["reciprocal_rank"], 0.5)
        self.assertEqual(scores["hit_rate"], 1.0)
        self.assertGreater(scores["ndcg"], 0.0)

    def test_duplicate_retrievals_only_count_once(self):
        scores = retrieval_metrics({"a", "b"}, ["a", "a"], k=2)
        self.assertEqual(scores["precision"], 0.5)
        self.assertEqual(scores["recall"], 0.5)

    def test_invalid_cutoff(self):
        with self.assertRaises(ValueError):
            retrieval_metrics({"a"}, ["a"], k=0)


class AnswerMetricsTests(unittest.TestCase):
    def test_context_relevance_uses_existing_result_shape(self):
        sources = [{"document": "Active learning selects informative examples."}]
        scores = context_relevance("How does active learning select examples?", sources)
        self.assertAlmostEqual(scores["context_relevance"], 0.75)

    def test_faithfulness_distinguishes_supported_and_unsupported_claims(self):
        scores = faithfulness(
            "Active learning selects informative examples. It cures every disease.",
            ["Active learning selects informative examples from an unlabeled pool."],
        )
        self.assertEqual(scores["faithfulness"], 0.5)
        self.assertEqual(scores["supported_claims"], 1.0)

    def test_attribution_validity_and_coverage(self):
        scores = source_attribution("Finding is reported [Source 1]. Another claim [4].", ["src"])
        self.assertEqual(scores["citation_validity"], 0.5)
        self.assertEqual(scores["citation_coverage"], 1.0)

    def test_evaluate_pipeline_shape_and_optional_retrieval(self):
        sources = [
            {"id": "chunk-a", "document": "Active learning selects examples.", "metadata": {"chunk_id": "chunk-a"}},
            {"id": "chunk-b", "document": "Unrelated material about geology.", "metadata": {"chunk_id": "chunk-b"}},
        ]
        scores = evaluate_rag(
            "How does active learning select examples?",
            "Active learning selects examples [1].",
            sources,
            relevant_ids={"chunk-a"},
            k=2,
        )
        self.assertIn("faithfulness", scores)
        self.assertIn("citation_coverage", scores)
        self.assertEqual(scores["retrieval"]["reciprocal_rank"], 1.0)


if __name__ == "__main__":
    unittest.main()
