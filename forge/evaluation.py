"""Evaluation system: prompt generation, output scoring, and round comparison."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Optional

from forge.session import EvalRun, SessionManager

if TYPE_CHECKING:
    from forge.providers.base import BaseProvider
    from forge.training import ForgeTrainer

logger = logging.getLogger(__name__)

# Scoring dimensions and their weights
SCORE_DIMENSIONS = ["relevance", "coherence", "instruction_following", "factual_accuracy"]

EVAL_PROMPT_GENERATION_TEMPLATE = """\
You are generating diverse test prompts for evaluating a language model trained on this dataset: {dataset_source}.

Generate exactly {num_prompts} diverse test prompts that:
1. Cover a wide range of topics related to the training domain
2. Include both simple and complex requests
3. Test different skills: factual Q&A, instruction following, reasoning, etc.
{category_hint}

Respond with ONLY a JSON array of strings, no other text:
["prompt 1", "prompt 2", ...]
"""

EVAL_SCORING_TEMPLATE = """\
You are evaluating the quality of language model outputs for a model fine-tuned on {dataset_source}.

For each prompt-response pair below, score the response on a 1-5 scale for each dimension:
- relevance: Does the response address the prompt?
- coherence: Is the response coherent and well-structured?
- instruction_following: Does the response follow the instructions?
- factual_accuracy: Is the content accurate and reasonable?

Prompt-Response Pairs:
{pairs_json}

Respond with ONLY a JSON object like this (do not include any other text):
{{
  "scores": [
    {{"relevance": 4, "coherence": 5, "instruction_following": 4, "factual_accuracy": 3}},
    ...
  ],
  "summary": "Brief overall assessment"
}}
"""


class EvalResult:
    def __init__(
        self,
        avg_score: float,
        scores: dict[str, float],
        passed_threshold: bool,
        summary: str,
        raw_scores: list[dict[str, int]],
        prompts: list[str],
        responses: list[str],
    ) -> None:
        self.avg_score = avg_score
        self.scores = scores
        self.passed_threshold = passed_threshold
        self.summary = summary
        self.raw_scores = raw_scores
        self.prompts = prompts
        self.responses = responses


class Evaluator:
    """Runs evaluation by generating prompts, getting model outputs, and scoring them."""

    def __init__(
        self,
        provider: "BaseProvider",
        trainer: "ForgeTrainer",
        session: SessionManager,
        min_quality_score: float = 3.0,
    ) -> None:
        self.provider = provider
        self.trainer = trainer
        self.session = session
        self.min_quality_score = min_quality_score

    def run(
        self,
        num_prompts: int = 15,
        categories: Optional[list[str]] = None,
    ) -> EvalResult:
        """Synchronous entry point — runs the async eval pipeline."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, self._run_async(num_prompts, categories))
                    return future.result()
            return loop.run_until_complete(self._run_async(num_prompts, categories))
        except RuntimeError:
            return asyncio.run(self._run_async(num_prompts, categories))

    async def _run_async(
        self,
        num_prompts: int,
        categories: Optional[list[str]],
    ) -> EvalResult:
        dataset_source = self.session.state.dataset_source
        round_num = self.session.state.current_round

        logger.info(
            "Starting eval: %d prompts, round %d, dataset=%s",
            num_prompts, round_num, dataset_source,
        )

        # Step 1: Generate test prompts via LLM
        prompts = await self._generate_prompts(
            dataset_source=dataset_source,
            num_prompts=num_prompts,
            categories=categories or [],
        )
        logger.info("Generated %d eval prompts", len(prompts))

        # Step 2: Get model outputs
        responses = self.trainer.generate(prompts)
        logger.info("Got %d model responses", len(responses))

        # Step 3: Score via LLM
        raw_scores, summary = await self._score_outputs(
            dataset_source=dataset_source,
            prompts=prompts,
            responses=responses,
        )

        # Compute aggregate scores
        dim_avgs: dict[str, float] = {}
        for dim in SCORE_DIMENSIONS:
            vals = [s.get(dim, 3) for s in raw_scores]
            dim_avgs[dim] = round(sum(vals) / len(vals), 2) if vals else 3.0

        avg_score = round(sum(dim_avgs.values()) / len(dim_avgs), 2)
        passed = avg_score >= self.min_quality_score

        result = EvalResult(
            avg_score=avg_score,
            scores=dim_avgs,
            passed_threshold=passed,
            summary=summary,
            raw_scores=raw_scores,
            prompts=prompts,
            responses=responses,
        )

        # Save to session
        eval_run = EvalRun(
            timestamp=_now(),
            round_num=round_num,
            avg_score=avg_score,
            scores=dim_avgs,
            passed_threshold=passed,
            summary=summary,
        )
        self.session.add_eval_result(eval_run)

        logger.info(
            "Eval complete: avg_score=%.2f, passed=%s, summary=%s",
            avg_score, passed, summary[:100],
        )

        if not passed:
            msg = (
                f"Evaluation score {avg_score:.2f} is below threshold {self.min_quality_score}. "
                f"Weakest dimension: {_weakest(dim_avgs)}. "
                f"Agent summary: {summary}"
            )
            logger.warning(msg)

        return result

    async def _generate_prompts(
        self,
        dataset_source: str,
        num_prompts: int,
        categories: list[str],
    ) -> list[str]:
        category_hint = ""
        if categories:
            category_hint = f"Focus on these categories: {', '.join(categories)}."

        user_content = EVAL_PROMPT_GENERATION_TEMPLATE.format(
            dataset_source=dataset_source,
            num_prompts=num_prompts,
            category_hint=category_hint,
        )

        response = await self.provider.chat(
            messages=[{"role": "user", "content": user_content}]
        )

        text = (response.content or "").strip()
        try:
            prompts = json.loads(text)
            if isinstance(prompts, list):
                return [str(p) for p in prompts[:num_prompts]]
        except json.JSONDecodeError:
            pass

        # Fallback: extract lines that look like prompts
        lines = [l.strip().strip('"').strip("'") for l in text.splitlines() if l.strip()]
        prompts = [l for l in lines if len(l) > 10][:num_prompts]
        if not prompts:
            # Last resort: generic prompts
            logger.warning("Could not parse LLM-generated prompts, using generic fallback")
            prompts = [
                f"Explain the concept of {dataset_source} in simple terms.",
                "What are the key principles in this domain?",
                "Give me an example of a common task in this area.",
            ][:num_prompts]
        return prompts

    async def _score_outputs(
        self,
        dataset_source: str,
        prompts: list[str],
        responses: list[str],
    ) -> tuple[list[dict[str, int]], str]:
        pairs = [
            {"prompt": p, "response": r}
            for p, r in zip(prompts, responses)
        ]

        user_content = EVAL_SCORING_TEMPLATE.format(
            dataset_source=dataset_source,
            pairs_json=json.dumps(pairs, ensure_ascii=False),
        )

        response = await self.provider.chat(
            messages=[{"role": "user", "content": user_content}]
        )

        text = (response.content or "").strip()
        try:
            obj = json.loads(text)
            raw_scores = obj.get("scores", [])
            summary = obj.get("summary", "No summary provided")
            # Validate structure
            validated: list[dict[str, int]] = []
            for s in raw_scores:
                entry: dict[str, int] = {}
                for dim in SCORE_DIMENSIONS:
                    entry[dim] = max(1, min(5, int(s.get(dim, 3))))
                validated.append(entry)
            return validated, summary
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("Could not parse scoring response: %s", text[:200])
            # Return neutral scores
            neutral = [{dim: 3 for dim in SCORE_DIMENSIONS} for _ in prompts]
            return neutral, "Scoring parse error — defaulted to neutral scores."

    def compare_with_previous(self) -> Optional[dict[str, Any]]:
        """Return a delta comparison with the previous eval round, if available."""
        results = self.session.state.eval_results
        if len(results) < 2:
            return None
        prev = results[-2]
        curr = results[-1]
        delta = round(curr.avg_score - prev.avg_score, 2)
        return {
            "previous_round": prev.round_num,
            "previous_score": prev.avg_score,
            "current_round": curr.round_num,
            "current_score": curr.avg_score,
            "delta": delta,
            "improved": delta > 0,
        }


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _weakest(scores: dict[str, float]) -> str:
    if not scores:
        return "unknown"
    return min(scores, key=scores.__getitem__)
