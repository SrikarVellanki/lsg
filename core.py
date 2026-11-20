"""Core logic for the Legal Sentence Game.

One section per milestone:

1. Corpus preparation        -> clean_text, tokenize, prepare_corpus
2. N-gram next-word predictor -> NGramModel
3. GPT-2 clause classifier    -> ClauseClassifier, load_classifier
4. (CLI lives in app/cli.py, it uses the helpers at the bottom of this file)
5. Feedback                   -> build_feedback, score_result
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field

from app import corpus

SENT_START = "<s>"
SENT_END = "</s>"
SPECIAL_TOKENS = {SENT_START, SENT_END}
BACKOFF = 0.4  # "stupid backoff" penalty per step down in n-gram order

# =============================================================================
# Milestone 1 - Prepare a legal text corpus
# =============================================================================

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[.,;:()]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def clean_text(text: str) -> str:
    """Normalise quotes/dashes and collapse whitespace."""
    text = (
        text.replace("\u2019", "'").replace("\u2018", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2013", "-").replace("\u2014", "-")
    )
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Lower-case word tokenizer that keeps ``. , ; : ( )`` as their own tokens."""
    return _TOKEN_RE.findall(clean_text(text).lower())


def split_sentences(text: str) -> list[str]:
    cleaned = clean_text(text)
    return [s for s in _SENTENCE_SPLIT_RE.split(cleaned) if s]


def prepare_corpus(text: str | list[str] | None = None) -> list[list[str]]:
    """Turn raw legal text into a list of token lists (one per sentence).

    ``text`` may be a single string, a list of sentences, or ``None`` to use the
    built-in corpus from :mod:`app.corpus`.
    """
    if text is None:
        sentences = corpus.all_sentences()
    elif isinstance(text, str):
        sentences = split_sentences(text)
    else:
        sentences = list(text)

    prepared = []
    for sentence in sentences:
        tokens = tokenize(sentence)
        if len(tokens) >= 3:  # drop fragments such as stray headings
            prepared.append(tokens)
    return prepared


# =============================================================================
# Milestone 2 - N-gram next-word predictor
# =============================================================================


class NGramModel:
    """Word n-gram model with "stupid backoff" for unseen contexts.

    P(w | context) = count(context + w) / count(context). If that n-gram was never seen we
    back off to a shorter context and multiply by ``BACKOFF`` (0.4).
    """

    def __init__(self, n: int = 3):
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        self.counts: dict[int, Counter] = {k: Counter() for k in range(1, n + 1)}
        self.context_counts: dict[int, Counter] = {k: Counter() for k in range(2, n + 1)}
        self.total_unigrams = 0
        self.vocab: set[str] = set()

    # ------------------------------------------------------------------ training
    def fit(self, sentences: list[list[str]]) -> "NGramModel":
        for tokens in sentences:
            padded = [SENT_START] * (self.n - 1) + list(tokens) + [SENT_END]
            self.vocab.update(tokens)
            for k in range(1, self.n + 1):
                for i in range(len(padded) - k + 1):
                    gram = tuple(padded[i : i + k])
                    self.counts[k][gram] += 1
                    if k > 1:
                        self.context_counts[k][gram[:-1]] += 1
        self.total_unigrams = sum(self.counts[1].values())
        return self

    # ------------------------------------------------------------------ probabilities
    def _context(self, tokens: list[str]) -> tuple[str, ...]:
        if self.n == 1:
            return ()
        padded = [SENT_START] * (self.n - 1) + list(tokens)
        return tuple(padded[-(self.n - 1):])

    def probability(self, word: str, tokens: list[str]) -> float:
        """Backoff probability of ``word`` following ``tokens``."""
        ctx = self._context(tokens)
        penalty = 1.0
        while ctx:
            k = len(ctx) + 1
            denom = self.context_counts[k].get(ctx, 0)
            num = self.counts[k].get(ctx + (word,), 0)
            if denom and num:
                return penalty * num / denom
            ctx = ctx[1:]
            penalty *= BACKOFF
        if not self.total_unigrams:
            return 0.0
        return penalty * self.counts[1].get((word,), 0) / self.total_unigrams

    def predict_next(self, tokens: list[str], k: int = 3) -> list[tuple[str, float]]:
        """Top-``k`` most likely next words as ``(word, probability)`` pairs."""
        scored = [
            (word, self.probability(word, tokens))
            for word in self.vocab
            if word not in SPECIAL_TOKENS
        ]
        scored = [item for item in scored if item[1] > 0]
        scored.sort(key=lambda item: (-item[1], item[0]))  # deterministic ordering
        return scored[:k]

    def perplexity(self, sentences: list[list[str]]) -> float:
        """Rough quality check (lower is better). Uses the backoff scores, so it is not a true probability."""
        log_sum, count = 0.0, 0
        for tokens in sentences:
            history: list[str] = []
            for word in tokens + [SENT_END]:
                p = self.probability(word, history) or 1e-9
                log_sum += math.log(p)
                count += 1
                history.append(word)
        return math.exp(-log_sum / count) if count else float("inf")


def build_ngram_model(n: int = 3, text: str | list[str] | None = None) -> NGramModel:
    return NGramModel(n).fit(prepare_corpus(text))


# =============================================================================
# Milestone 3 - Fine-tuned GPT-2 clause classifier
# =============================================================================

# Used only when no fine-tuned model is available, so the game still runs.
_KEYWORDS: dict[str, list[str]] = {
    "Confidentiality": ["confidential", "disclos", "proprietary", "secrecy", "need to know", "receiving party"],
    "Termination": ["terminat", "expir", "cure", "cease", "surviv", "gross misconduct"],
    "Indemnification": ["indemn", "hold harmless", "defend", "defence", "harmless"],
    "Governing Law": ["governed", "governing", "laws of", "jurisdiction", "courts", "construed", "choice of law"],
    "Payment": ["pay", "invoice", "fees", "interest", "purchase price", "rent", "reimburs", "tax"],
    "Limitation of Liability": ["liable", "liability", "consequential", "indirect", "aggregate", "exceed", "no event", "under no circumstances"],
    "Force Majeure": ["force majeure", "beyond", "act of god", "acts of god", "natural disaster", "pandemic", "flood", "unforeseeable", "outside its control"],
    "Assignment": ["assign", "transfer", "successor", "subcontract", "delegate", "sublicense", "change of control"],
    "Warranty": ["warrant", "represent", "as is", "defect", "merchantab", "title to"],
}


@dataclass
class ClauseResult:
    label: str
    confidence: float
    backend: str  # "gpt2" or "keyword"
    scores: dict[str, float] = field(default_factory=dict)

    def top(self, k: int = 3) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda kv: -kv[1])[:k]


class ClauseClassifier:
    """Classifies a sentence into a legal clause type.

    Uses the fine-tuned GPT-2 model saved by ``python -m scripts.train_classifier`` when it
    exists in ``model_dir``; otherwise falls back to a simple keyword scorer so that the game
    (and the test-suite) work on any machine.
    """

    def __init__(self, model_dir: str | None = None, allow_fallback: bool = True):
        self.model_dir = model_dir
        self.backend = "keyword"
        self.load_error: str | None = None
        self._tokenizer = None
        self._model = None
        self._torch = None

        if model_dir and os.path.isdir(model_dir):
            try:
                self._load_gpt2(model_dir)
            except Exception as exc:  # missing torch/transformers, corrupt files, ...
                self.load_error = f"{type(exc).__name__}: {exc}"
                if not allow_fallback:
                    raise
        elif model_dir and not allow_fallback:
            raise FileNotFoundError(f"No fine-tuned model found in {model_dir!r}")

    # ------------------------------------------------------------------ GPT-2 backend
    def _load_gpt2(self, model_dir: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        self._torch, self._tokenizer, self._model = torch, tokenizer, model
        self.backend = "gpt2"

    def _classify_gpt2(self, sentence: str) -> ClauseResult:
        torch = self._torch
        inputs = self._tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = self._model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).tolist()
        id2label = self._model.config.id2label
        scores = {id2label[i]: float(p) for i, p in enumerate(probs)}
        label = max(scores, key=scores.get)
        return ClauseResult(label, scores[label], "gpt2", scores)

    # ------------------------------------------------------------------ keyword backend
    @staticmethod
    def _classify_keywords(sentence: str) -> ClauseResult:
        lowered = clean_text(sentence).lower()
        raw = {label: sum(kw in lowered for kw in kws) for label, kws in _KEYWORDS.items()}
        total = sum(raw.values())
        if total == 0:
            return ClauseResult("Unclear", 0.0, "keyword", {})
        scores = {label: hits / total for label, hits in raw.items()}
        label = max(scores, key=scores.get)
        return ClauseResult(label, scores[label], "keyword", scores)

    # ------------------------------------------------------------------ public API
    def classify(self, sentence: str) -> ClauseResult:
        if not sentence.strip():
            return ClauseResult("Unclear", 0.0, self.backend, {})
        if self.backend == "gpt2":
            return self._classify_gpt2(sentence)
        return self._classify_keywords(sentence)


def load_classifier(model_dir: str | None = None) -> ClauseClassifier:
    """Load the fine-tuned GPT-2 classifier (falls back to keywords if it is not trained yet)."""
    if model_dir is None:
        from app.config import get_settings

        model_dir = get_settings().classifier_dir
    return ClauseClassifier(model_dir)


# =============================================================================
# Milestone 5 - Feedback after the player finishes a sentence
# =============================================================================


def score_result(target: str, result: ClauseResult) -> int:
    """10 points for the intended clause type (+ up to 5 for confidence), 2 for trying, 0 if empty."""
    if result.label == "Unclear":
        return 0
    if result.label == target:
        return 10 + round(5 * result.confidence)
    return 2


def build_feedback(target: str, sentence: str, result: ClauseResult) -> str:
    """Human-readable feedback for one finished sentence."""
    lines = [f"Your sentence: {sentence}", ""]
    if result.label == "Unclear":
        lines.append("Classifier: I could not recognise a clause type in that sentence.")
        lines.append(f"Tip: a {target} clause usually uses terms such as "
                     f"{', '.join(repr(k) for k in _KEYWORDS.get(target, [])[:3])}.")
        return "\n".join(lines)

    verdict = "match" if result.label == target else "mismatch"
    lines.append(f"Classifier ({result.backend}): {result.label} ({result.confidence:.0%} confident)")
    lines.append(f"Target clause type:  {target} -> {verdict}")

    others = [(l, s) for l, s in result.top(3) if l != result.label and s > 0]
    if others:
        lines.append("Also considered: " + ", ".join(f"{l} ({s:.0%})" for l, s in others))

    if result.label == target:
        lines.append("Nice drafting - your sentence reads like a " + target + " clause.")
    else:
        hints = ", ".join(repr(k) for k in _KEYWORDS.get(target, [])[:4])
        lines.append(f"It reads more like {result.label}. To steer it towards {target}, "
                     f"try wording such as {hints}.")

    word_count = len(tokenize(sentence))
    if word_count < 8:
        lines.append("Heads-up: real clauses are usually longer and more specific than this.")
    if not sentence.rstrip().endswith("."):
        lines.append("Heads-up: remember to end the sentence with a full stop.")
    return "\n".join(lines)


# =============================================================================
# Helpers shared with the CLI (Milestone 4)
# =============================================================================


def join_text(text: str, piece: str) -> str:
    """Append a word or punctuation mark to a sentence with sensible spacing."""
    if not text:
        return piece
    if piece in {".", ",", ";", ":", ")"}:
        return text + piece
    if text.endswith("("):
        return text + piece
    return f"{text} {piece}"


def sentence_is_complete(text: str) -> bool:
    return text.rstrip().endswith(".")
