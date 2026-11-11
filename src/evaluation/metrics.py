"""
RAG Evaluation Metrics Module
Evaluates: Retrieval Relevance, Answer Faithfulness, Context Relevance, and Source Attribution.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import re


class BaseMetric(ABC):
    """Abstract base class for all RAG evaluation metrics."""

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_contexts: List[str],
        answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        pass


class RetrievalRelevanceMetric(BaseMetric):
    """
    Evaluates whether the retrieved contexts are relevant to the user query.
    Calculates keyword overlap / token relevance or semantic alignment ratio.
    """

    def evaluate(
        self,
        query: str,
        retrieved_contexts: List[str],
        answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not retrieved_contexts:
            return {"score": 0.0, "reason": "No retrieved contexts provided."}

        query_tokens = set(re.findall(r"\w+", query.lower()))
        # Filter out common stop words if desired, keeping core query terms
        stop_words = {"what", "is", "the", "a", "an", "how", "to", "in", "of", "for", "and"}
        core_query_tokens = query_tokens - stop_words
        if not core_query_tokens:
            core_query_tokens = query_tokens

        scores = []
        for ctx in retrieved_contexts:
            ctx_tokens = set(re.findall(r"\w+", ctx.lower()))
            overlap = len(core_query_tokens.intersection(ctx_tokens))
            score = min(1.0, overlap / max(1, len(core_query_tokens)))
            scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return {
            "score": round(avg_score, 4),
            "reason": f"Average token overlap across {len(retrieved_contexts)} context chunks.",
        }


class AnswerFaithfulnessMetric(BaseMetric):
    """
    Evaluates whether the generated answer is strictly grounded in the retrieved contexts
    (detects hallucinations / unsupported claims).
    """

    def evaluate(
        self,
        query: str,
        retrieved_contexts: List[str],
        answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not answer.strip():
            return {"score": 0.0, "reason": "Empty answer provided."}
        if not retrieved_contexts:
            return {"score": 0.0, "reason": "No retrieved context available to anchor faithfulness."}

        # Decompose answer into sentence/claim segments
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        if not sentences:
            return {"score": 1.0, "reason": "No substantive claims found."}

        combined_context = " ".join(retrieved_contexts).lower()
        supported_count = 0

        for sentence in sentences:
            # Extract key terms from sentence
            words = [w for w in re.findall(r"\w+", sentence.lower()) if len(w) > 3]
            if not words:
                supported_count += 1
                continue
            
            # Check how many key words appear in the combined context
            matched = sum(1 for w in words if w in combined_context)
            ratio = matched / len(words)
            if ratio >= 0.5:  # threshold for claim support
                supported_count += 1

        faithfulness_score = supported_count / len(sentences)
        return {
            "score": round(faithfulness_score, 4),
            "reason": f"{supported_count}/{len(sentences)} claims verified against retrieved contexts.",
        }


class ContextRelevanceMetric(BaseMetric):
    """
    Evaluates context precision / signal-to-noise ratio in the retrieved passages relative to the query.
    """

    def evaluate(
        self,
        query: str,
        retrieved_contexts: List[str],
        answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not retrieved_contexts:
            return {"score": 0.0, "reason": "No contexts retrieved."}

        query_words = set(re.findall(r"\w+", query.lower()))
        relevant_chunks = 0

        for ctx in retrieved_contexts:
            ctx_words = set(re.findall(r"\w+", ctx.lower()))
            # If intersection has meaningful overlap
            if len(query_words.intersection(ctx_words)) > 0:
                relevant_chunks += 1

        score = relevant_chunks / len(retrieved_contexts)
        return {
            "score": round(score, 4),
            "reason": f"{relevant_chunks} out of {len(retrieved_contexts)} retrieved chunks contain query terms.",
        }


class SourceAttributionMetric(BaseMetric):
    """
    Evaluates whether generated citations or source references accurately link back to the provided source metadata.
    """

    def evaluate(
        self,
        query: str,
        retrieved_contexts: List[str],
        answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not sources:
            # If system doesn't require explicit structured sources or none provided
            return {"score": 1.0, "reason": "No explicit source metadata provided for validation."}

        # Check if citations in answer match available source identifiers/titles
        answer_lower = answer.lower()
        matched_sources = 0
        total_sources = len(sources)

        if total_sources == 0:
            return {"score": 1.0, "reason": "Empty source list."}

        for src in sources:
            src_name = str(src.get("title", src.get("id", ""))).lower()
            if src_name and src_name in answer_lower:
                matched_sources += 1

        # Fallback proportion of source utilization/attribution mention
        score = max(0.5, matched_sources / total_sources) if matched_sources > 0 else 0.5
        return {
            "score": round(score, 4),
            "reason": f"Matched {matched_sources}/{total_sources} source references in the generated output.",
        }