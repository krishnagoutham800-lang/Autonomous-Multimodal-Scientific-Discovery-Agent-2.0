from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.metrics import (
    AnswerFaithfulnessMetric,
    ContextRelevanceMetric,
    RetrievalRelevanceMetric,
    SourceAttributionMetric,
)


def retrieval_relevance(*args, **kwargs):
    return RetrievalRelevanceMetric().evaluate(*args, **kwargs)


def answer_faithfulness(*args, **kwargs):
    return AnswerFaithfulnessMetric().evaluate(*args, **kwargs)


def context_relevance(*args, **kwargs):
    return ContextRelevanceMetric().evaluate(*args, **kwargs)


def source_attribution(*args, **kwargs):
    return SourceAttributionMetric().evaluate(*args, **kwargs)


__all__ = [
    "RAGEvaluator",
    "RetrievalRelevanceMetric",
    "AnswerFaithfulnessMetric",
    "ContextRelevanceMetric",
    "SourceAttributionMetric",
    "retrieval_relevance",
    "answer_faithfulness",
    "context_relevance",
    "source_attribution",
]