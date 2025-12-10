"""
Evaluation framework using DeepEval metrics.
"""
import logging
from typing import List, Dict, Optional, Any, Callable
import json
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)

# DeepEval 
try:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        HallucinationMetric,
        SummarizationMetric,
    )
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    logging.warning("DeepEval not available. Install with: pip install deepeval")



# RAG Evaluator
class RAGEvaluator:
    """Evaluate RAG system performance using DeepEval metrics."""

    def __init__(self, use_ragas: bool = False, use_deepeval: bool = True,
                 llm_model: Optional[str] = None):
        self.use_ragas = False
        self.use_deepeval = use_deepeval and DEEPEVAL_AVAILABLE
        self.llm_model = llm_model or "gpt-4o-mini"
        self._executor = ThreadPoolExecutor(max_workers=4)

        if not self.use_ragas and not self.use_deepeval:
            logger.warning("No evaluation frameworks enabled.")

    def _run_in_thread(self, func: Callable, *args, **kwargs):
        future = self._executor.submit(func, *args, **kwargs)
        return future.result()

    
    # Public API
    def evaluate_rrag_contexts(self, contexts):
        """
        Ensures contexts are a list of plain strings (required by RAGAS).
        """
        clean = []
        for c in contexts:
            if isinstance(c, dict):
                if "text" in c:
                    clean.append(c["text"])
                else:
                    clean.append(str(c))
            elif isinstance(c, str):
                clean.append(c)
            else:
                clean.append(str(c))
        return clean

    def evaluate_rag_response(
        self, query: str, answer: str, contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict:

        clean_contexts = self.evaluate_rrag_contexts(contexts)

        results = {
            "query": query,
            "answer": answer,
            "num_contexts": len(clean_contexts)
        }

        # DeepEval
        if self.use_deepeval:
            try:
                results["deepeval"] = self._evaluate_with_deepeval(
                    query, answer, clean_contexts, ground_truth
                )
            except Exception as e:
                logger.exception("DeepEval evaluation error")
                results["deepeval"] = {"error": str(e)}
        #  Overall score 
        results["overall_score"] = self._calculate_overall_score(results)
        return results

    
    # DeepEval Evaluation
    def _evaluate_with_deepeval(
        self, query: str, answer: str, contexts: List[str], ground_truth: Optional[str]
    ) -> Dict:

        if not DEEPEVAL_AVAILABLE:
            return {"error": "DeepEval not installed"}

        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=contexts,
            context=contexts,
            expected_output=ground_truth
        )

        metrics = self._build_deepeval_metrics(bool(ground_truth))

        results = {}

        def run_metric(metric, test_case):
            """
            Runs DeepEval metric inside a fresh event loop.
            Avoids uvloop conflict from FastAPI/Uvicorn.
            """
            async def _exec():
                return metric.measure(test_case)
            return asyncio.run(_exec())

        for metric in metrics:
            try:
                self._run_in_thread(run_metric, metric, test_case)

                results[metric.__class__.__name__] = {
                    "score": metric.score,
                    "reason": metric.reason
                }
            except Exception as e:
                results[metric.__class__.__name__] = {"error": str(e)}

        return results

    def _build_deepeval_metrics(self, has_ground_truth: bool) -> List:
        """
        Construct DeepEval metrics, adding reference-dependent ones only when ground truth is provided.
        """
        base_kwargs = {"threshold": 0.7, "model": self.llm_model}
        metrics: List = [
            AnswerRelevancyMetric(**base_kwargs),
            FaithfulnessMetric(**base_kwargs),
            ContextualPrecisionMetric(**base_kwargs),
            ContextualRelevancyMetric(**base_kwargs),
            HallucinationMetric(**base_kwargs),
        ]

        if has_ground_truth:
            metrics.extend([
                ContextualRecallMetric(**base_kwargs),
                SummarizationMetric(**base_kwargs),
            ])

        return metrics

    
    # Overall Score
    def _calculate_overall_score(self, results: Dict) -> Optional[float]:
        vals = []

        # DeepEval
        d = results.get("deepeval", {})
        if isinstance(d, dict):
            for k, obj in d.items():
                if isinstance(obj, dict) and "score" in obj:
                    vals.append(obj["score"])

        return sum(vals) / len(vals) if vals else None

    
    # Batch Evaluation
    def batch_evaluate(self, test_cases: List[Dict]) -> Dict:
        all_res = []

        for i, tc in enumerate(test_cases):
            logger.info(f"Evaluating case {i+1}/{len(test_cases)}")
            all_res.append(
                self.evaluate_rag_response(
                    tc["query"],
                    tc["answer"],
                    tc["retrieved_contexts"],
                    tc.get("ground_truth")
                )
            )

        return {
            "num_test_cases": len(test_cases),
            "individual_results": all_res,
            "aggregate_metrics": self._aggregate(all_res)
        }

    def _aggregate(self, results: List[Dict]) -> Dict:
        agg = {"overall_scores": [], "deepeval": {}}

        for r in results:

            if r.get("overall_score") is not None:
                agg["overall_scores"].append(r["overall_score"])

            # DeepEval
            if "deepeval" in r:
                for k, obj in r["deepeval"].items():
                    if isinstance(obj, dict) and "score" in obj:
                        agg["deepeval"].setdefault(k, []).append(obj["score"])

        return {
            "average_overall_score":
                (sum(agg["overall_scores"]) / len(agg["overall_scores"])
                 if agg["overall_scores"] else None),
            "deepeval_avg": {k: sum(v)/len(v) for k, v in agg["deepeval"].items()}
        }



# User Feedback Collector
class UserFeedbackCollector:

    def __init__(self, feedback_file: str = "./data/feedback.json"):
        self.feedback_file = feedback_file
        self._ensure_file()

    def _ensure_file(self):
        import os
        from pathlib import Path
        Path(self.feedback_file).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, "w") as f:
                json.dump([], f)

    def add_feedback(self, feedback: Dict) -> bool:
        try:
            with open(self.feedback_file, "r") as f:
                all_fb = json.load(f)

            from datetime import datetime
            feedback["timestamp"] = datetime.now().isoformat()
            all_fb.append(feedback)

            with open(self.feedback_file, "w") as f:
                json.dump(all_fb, f, indent=2)

            return True
        except Exception as e:
            logger.error(f"Error saving feedback: {str(e)}")
            return False

    def get_feedback_summary(self) -> Dict:
        try:
            with open(self.feedback_file, "r") as f:
                all_fb = json.load(f)

            if not all_fb:
                return {"total_feedback": 0}

            ratings = [f.get("rating") for f in all_fb if isinstance(f.get("rating"), (int, float))]

            return {
                "total_feedback": len(all_fb),
                "average_rating": sum(ratings) / len(ratings) if ratings else None,
                "rating_distribution": self._dist(ratings),
                "recent_feedback": all_fb[-5:]
            }

        except Exception as e:
            return {"error": str(e)}

    def _dist(self, ratings: List[int]) -> Dict:
        dist = {i: 0 for i in range(1, 6)}
        for r in ratings:
            if r in dist:
                dist[r] += 1
        return dist
