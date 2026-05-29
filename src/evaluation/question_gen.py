"""Question generation for the RAG Comparison System evaluation.

Generates candidate evaluation questions with ground-truth answers from the
arXiv paper corpus. Each paper is sent to the LLM which returns a JSON array
of question/ground_truth/source_pages objects. Results are saved to
data/questions.json.

Idempotent: if questions.json already exists and contains >= count questions,
generation is skipped entirely.

Cost gate: the generate_candidates() method accepts an approved flag. When
False (the default), it prints a cost estimate and returns an empty list.
When True, it executes the LLM calls.

Typical usage (from CLI):
    from src.evaluation.question_gen import QuestionGenerator
    gen = QuestionGenerator(provider, config)
    questions = gen.generate_candidates(documents_dir, count=30, approved=True)
    gen.save(questions, "data/questions.json")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.cost_gate import CostGate

if TYPE_CHECKING:
    from src.config import AppConfig
    from src.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EvalQuestion dataclass
# ---------------------------------------------------------------------------


@dataclass
class EvalQuestion:
    """A single evaluation question with ground truth and source provenance."""

    question_id: str        # Deterministic SHA-256 of the question text (first 16 hex chars)
    question: str           # The evaluation question
    ground_truth: str       # 2-4 sentence answer with specific page citations
    source_files: list[str] = field(default_factory=list)   # Originating paper filenames
    source_pages: list[int] = field(default_factory=list)   # Cited page numbers


def _make_question_id(question_text: str) -> str:
    """Deterministic 16-hex-char ID from question text (SHA-256)."""
    return hashlib.sha256(question_text.strip().encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert research assistant that creates high-quality evaluation "
    "questions for RAG (Retrieval-Augmented Generation) system benchmarks. "
    "You produce well-formed JSON output only — no prose outside the JSON array."
)

_USER_PROMPT_TMPL = """\
Read the following research paper and generate {n_questions} evaluation questions.

Requirements for each question:
1. ANSWERABLE: the answer must be explicitly stated or strongly implied in the paper.
2. NON-TRIVIAL: requires understanding of concepts, methods, or results — not a simple keyword lookup.
3. DIVERSE: cover different aspects (motivation, methodology, experimental results, limitations, comparisons).
4. SPECIFIC: reference concrete details (numbers, model names, dataset names, technique names).

Requirements for each ground truth answer:
- 2-4 sentences long.
- Grounded in specific text from the paper.
- Include page number citations where the supporting evidence appears.
- Factually accurate and complete (a good RAG answer should match this).

Return ONLY a valid JSON array. No markdown fences, no prose, no commentary.
Each element must have exactly these fields:
  "question"     : string — the evaluation question
  "ground_truth" : string — 2-4 sentence answer with page citation(s)
  "source_pages" : array of integers — page number(s) where the answer is found

---BEGIN PAPER: {source_file}---
{paper_text}
---END PAPER---

JSON array:"""


# ---------------------------------------------------------------------------
# QuestionGenerator
# ---------------------------------------------------------------------------


class QuestionGenerator:
    """Generates evaluation questions from extracted paper Markdown files."""

    def __init__(self, provider: "BaseLLMProvider", config: "AppConfig") -> None:
        self._provider = provider
        self._config = config
        logger.info(
            "QuestionGenerator initialised: provider=%s model=%s",
            provider.__class__.__name__,
            provider.get_model_name(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_candidates(
        self,
        documents_dir: str,
        count: int = 30,
        approved: bool = False,
    ) -> list[EvalQuestion]:
        """Generate *count* candidate questions from extracted .md files.

        Args:
            documents_dir: Directory containing extracted .md files
                           (typically data/papers/extracted/).
            count:         Target number of candidate questions to generate.
                           Distributed roughly evenly across papers (~4 per paper).
            approved:      If False, prints cost estimate and returns [].
                           If True, executes LLM calls.

        Returns:
            List of EvalQuestion objects. Empty list when not approved.
        """
        extracted_path = Path(documents_dir)
        md_files = sorted(extracted_path.glob("*.md"))

        if not md_files:
            logger.warning(
                "generate_candidates: no .md files found in %s — returning empty list",
                documents_dir,
            )
            return []

        num_papers = len(md_files)
        # Distribute questions across papers, at least 1 per paper
        questions_per_paper = max(1, round(count / num_papers))
        total_calls = num_papers
        avg_input_tokens = self._config.cost_estimation.avg_paper_tokens
        avg_output_tokens = (
            questions_per_paper * self._config.cost_estimation.question_output_tokens
        )

        logger.info(
            "generate_candidates: %d papers, %d questions_per_paper, "
            "%d total_calls, avg_input=%d avg_output=%d",
            num_papers, questions_per_paper, total_calls,
            avg_input_tokens, avg_output_tokens,
        )

        # Cost gate — estimate first, execute only when approved
        gate = CostGate(self._config, approved=approved)
        estimate = gate.estimate(
            operation="question_gen",
            num_items=total_calls,
            avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens,
        )
        gate.display_estimate(estimate)
        gate.require_approval(estimate)  # exits (sys.exit(0)) if not approved

        # -----------------------------------------------------------------
        # Execute LLM calls
        # -----------------------------------------------------------------
        all_questions: list[EvalQuestion] = []

        for md_file in md_files:
            paper_questions = self._generate_for_paper(
                md_file=md_file,
                n_questions=questions_per_paper,
            )
            all_questions.extend(paper_questions)
            logger.info(
                "generate_candidates: %s -> %d questions (running total: %d)",
                md_file.name, len(paper_questions), len(all_questions),
            )

        logger.info(
            "generate_candidates: complete — %d questions from %d papers",
            len(all_questions), num_papers,
        )
        return all_questions

    def save(self, questions: list[EvalQuestion], path: str) -> None:
        """Persist questions to a JSON file.

        Args:
            questions: List of EvalQuestion objects to save.
            path:      Destination file path (e.g. "data/questions.json").
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        serialised = [asdict(q) for q in questions]
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(serialised, fh, indent=2, ensure_ascii=False)

        logger.info("save: wrote %d questions to %s", len(questions), out_path)

    def load(self, path: str) -> list[EvalQuestion]:
        """Load questions from a JSON file.

        Args:
            path: Source file path (e.g. "data/questions.json").

        Returns:
            List of EvalQuestion objects.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Questions file not found: {in_path}")

        with in_path.open("r", encoding="utf-8") as fh:
            raw: list[dict] = json.load(fh)

        questions = [
            EvalQuestion(
                question_id=item["question_id"],
                question=item["question"],
                ground_truth=item["ground_truth"],
                source_files=item.get("source_files", []),
                source_pages=item.get("source_pages", []),
            )
            for item in raw
        ]
        logger.info("load: read %d questions from %s", len(questions), in_path)
        return questions

    # ------------------------------------------------------------------
    # Idempotency helper
    # ------------------------------------------------------------------

    def is_already_generated(self, path: str, min_count: int) -> bool:
        """Return True if *path* exists and contains at least *min_count* questions.

        Use this before calling generate_candidates() to skip generation when
        a sufficient questions.json already exists.

        Args:
            path:      Path to questions.json.
            min_count: Minimum number of questions required to consider complete.
        """
        try:
            questions = self.load(path)
            if len(questions) >= min_count:
                logger.info(
                    "is_already_generated: %s has %d questions (>= %d) — skipping generation",
                    path, len(questions), min_count,
                )
                return True
            logger.info(
                "is_already_generated: %s has %d questions (< %d) — will regenerate",
                path, len(questions), min_count,
            )
            return False
        except FileNotFoundError:
            logger.debug("is_already_generated: %s not found — will generate", path)
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_for_paper(
        self,
        md_file: Path,
        n_questions: int,
    ) -> list[EvalQuestion]:
        """Call the LLM for a single paper and parse the JSON response.

        Args:
            md_file:     Path to the extracted .md file.
            n_questions: Number of questions to request.

        Returns:
            List of EvalQuestion objects parsed from LLM output.
            Returns [] on parse failure (logs a warning).
        """
        paper_text = md_file.read_text(encoding="utf-8")
        source_file = md_file.name

        prompt = _USER_PROMPT_TMPL.format(
            n_questions=n_questions,
            source_file=source_file,
            paper_text=paper_text,
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        logger.debug(
            "_generate_for_paper: calling LLM for %s (n_questions=%d)",
            source_file, n_questions,
        )

        t_start = time.monotonic()
        try:
            response = self._provider.complete(
                messages=messages,
                temperature=0.2,   # slight non-zero for question diversity
                max_tokens=self._config.llm.max_tokens,
            )
        except Exception as exc:
            logger.error(
                "_generate_for_paper: LLM call failed for %s: %s", source_file, exc
            )
            return []

        latency_ms = (time.monotonic() - t_start) * 1000
        logger.debug(
            "_generate_for_paper: %s — latency=%.0fms input_tokens=%d output_tokens=%d cost=$%.6f",
            source_file, latency_ms,
            response.input_tokens, response.output_tokens, response.cost_usd,
        )

        # Record telemetry
        self._provider._record_call(
            response=response,
            operation="question_gen",
            pipeline=None,
        )

        # Parse JSON response
        return self._parse_llm_response(response.text, source_file)

    def _parse_llm_response(
        self,
        response_text: str,
        source_file: str,
    ) -> list[EvalQuestion]:
        """Parse LLM JSON output into EvalQuestion objects.

        Handles minor formatting issues (e.g. accidental markdown fences).
        Logs a warning and returns [] on parse failure.

        Args:
            response_text: Raw LLM output (expected to be a JSON array).
            source_file:   Source filename for provenance metadata.

        Returns:
            List of EvalQuestion objects.
        """
        # Strip accidental markdown code fences if present
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop first and last fence lines
            inner_lines = []
            in_fence = False
            for line in lines:
                if line.startswith("```") and not in_fence:
                    in_fence = True
                    continue
                if line.startswith("```") and in_fence:
                    break
                inner_lines.append(line)
            text = "\n".join(inner_lines).strip()

        try:
            raw_items: list[dict] = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(
                "_parse_llm_response: JSON parse failed for %s: %s — skipping paper",
                source_file, exc,
            )
            logger.debug("_parse_llm_response: raw response text: %s", response_text[:500])
            return []

        if not isinstance(raw_items, list):
            logger.warning(
                "_parse_llm_response: expected JSON array, got %s for %s — skipping",
                type(raw_items).__name__, source_file,
            )
            return []

        questions: list[EvalQuestion] = []
        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                logger.warning(
                    "_parse_llm_response: item %d is not a dict for %s — skipping item",
                    idx, source_file,
                )
                continue

            question_text = item.get("question", "").strip()
            ground_truth = item.get("ground_truth", "").strip()
            source_pages_raw = item.get("source_pages", [])

            if not question_text or not ground_truth:
                logger.warning(
                    "_parse_llm_response: item %d missing question or ground_truth for %s — skipping",
                    idx, source_file,
                )
                continue

            # Normalise source_pages to list[int]
            source_pages: list[int] = []
            if isinstance(source_pages_raw, list):
                for p in source_pages_raw:
                    try:
                        source_pages.append(int(p))
                    except (TypeError, ValueError):
                        pass
            elif isinstance(source_pages_raw, (int, float)):
                source_pages = [int(source_pages_raw)]

            question_id = _make_question_id(question_text)

            questions.append(
                EvalQuestion(
                    question_id=question_id,
                    question=question_text,
                    ground_truth=ground_truth,
                    source_files=[source_file],
                    source_pages=source_pages,
                )
            )

        logger.debug(
            "_parse_llm_response: parsed %d/%d items for %s",
            len(questions), len(raw_items), source_file,
        )
        return questions
