"""Command-line interface for the Legal Sentence Game.

Usage:
    python -m app.cli hello          # check that everything is wired up
    python -m app.cli play           # play the game
    python -m app.cli play --rounds 3 --suggestions 4 --seed 1
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Callable

from app import corpus
from app.config import get_settings
from app.core import (
    ClauseClassifier,
    NGramModel,
    build_feedback,
    build_ngram_model,
    join_text,
    load_classifier,
    score_result,
    sentence_is_complete,
    tokenize,
)

MAX_WORDS_PER_SENTENCE = 40

HELP_TEXT = """\
How to play
  1, 2, 3 ...   accept the numbered n-gram suggestion
  any words     type your own words (they are added to the sentence)
  undo          remove the last thing you added
  done          finish the sentence now and get feedback
  skip          skip this prompt
  quit          leave the game
A sentence also finishes automatically when you end it with a full stop (.)"""


# --------------------------------------------------------------------------- one round
def play_round(
    prompt: corpus.GamePrompt,
    model: NGramModel,
    classifier: ClauseClassifier,
    suggestions: int = 3,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., None] = print,
) -> tuple[str, int] | None:
    """Play one prompt. Returns ``(sentence, points)`` or ``None`` if skipped.

    Raises ``SystemExit`` when the player types ``quit`` (or input ends).
    """
    print_fn(f"\nClause type to draft: {prompt.clause_type}")
    text = prompt.starter
    history: list[str] = []

    while True:
        print_fn(f"\n  {text} ...")
        options = model.predict_next(tokenize(text), k=suggestions)
        for i, (word, prob) in enumerate(options, start=1):
            print_fn(f"    [{i}] {word}   ({prob:.0%})")

        try:
            raw = input_fn("> ").strip()
        except EOFError:
            raise SystemExit(0)
        command = raw.lower()

        if command == "quit":
            raise SystemExit(0)
        if command == "skip":
            return None
        if command == "help":
            print_fn(HELP_TEXT)
            continue
        if command == "undo":
            if history:
                text = history.pop()
            else:
                print_fn("Nothing to undo.")
            continue
        if command == "done":
            break
        if not raw:
            continue

        history.append(text)
        if command.isdigit() and 1 <= int(command) <= len(options):
            text = join_text(text, options[int(command) - 1][0])
        else:
            for piece in _split_user_input(raw):
                text = join_text(text, piece)

        if sentence_is_complete(text) or len(tokenize(text)) >= MAX_WORDS_PER_SENTENCE:
            break

    result = classifier.classify(text)
    print_fn("\n" + "-" * 60)
    print_fn(build_feedback(prompt.clause_type, text, result))
    points = score_result(prompt.clause_type, result)
    print_fn(f"Points this round: {points}")
    print_fn("-" * 60)
    return text, points


def _split_user_input(raw: str) -> list[str]:
    """Split typed text into words, keeping the player's own capitalisation."""
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[.,;:()]", raw)


# --------------------------------------------------------------------------- high score
def load_best_score(path: str) -> int:
    try:
        return int(json.loads(Path(path).read_text(encoding="utf-8")).get("best_score", 0))
    except (OSError, ValueError, AttributeError):
        return 0


def save_best_score(path: str, score: int) -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"best_score": score}), encoding="utf-8")
    except OSError:
        pass  # saving a high score is optional


# --------------------------------------------------------------------------- whole game
def run_game(
    rounds: int = 3,
    suggestions: int = 3,
    seed: int | None = None,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., None] = print,
    model: NGramModel | None = None,
    classifier: ClauseClassifier | None = None,
    data_path: str | None = None,
) -> int:
    settings = get_settings()
    model = model or build_ngram_model(settings.ngram_order)
    classifier = classifier or load_classifier(settings.classifier_dir)
    data_path = data_path if data_path is not None else settings.data_path

    print_fn("=" * 60)
    print_fn(" LEGAL SENTENCE GAME")
    print_fn("=" * 60)
    print_fn("Complete each partial sentence using n-gram suggestions or your own words.")
    print_fn("A language model then tells you which clause type you drafted.")
    if classifier.backend != "gpt2":
        print_fn("(Note: fine-tuned GPT-2 not found - using the keyword fallback. "
                 "Run `python -m scripts.train_classifier` to train it.)")
    print_fn("Type 'help' at any prompt for the commands.")

    rng = random.Random(seed)
    prompts = rng.sample(corpus.GAME_PROMPTS, k=min(rounds, len(corpus.GAME_PROMPTS)))

    total = 0
    played = 0
    try:
        for number, prompt in enumerate(prompts, start=1):
            print_fn(f"\n=== Round {number} of {len(prompts)} ===")
            outcome = play_round(prompt, model, classifier, suggestions, input_fn, print_fn)
            if outcome is not None:
                total += outcome[1]
                played += 1
    except SystemExit:
        print_fn("\nLeaving the game early.")

    best = load_best_score(data_path)
    print_fn(f"\nFinal score: {total}  ({played} sentence(s) drafted)")
    if total > best:
        print_fn("New personal best!")
        save_best_score(data_path, total)
    else:
        print_fn(f"Personal best: {best}")
    return total


# --------------------------------------------------------------------------- entry point
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="Legal Sentence Game")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that the project runs")
    play = sub.add_parser("play", help="play the game")
    play.add_argument("--rounds", type=int, default=3, help="number of prompts (default 3)")
    play.add_argument("--suggestions", type=int, default=3, help="n-gram suggestions shown (default 3)")
    play.add_argument("--seed", type=int, default=None, help="random seed for repeatable games")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "hello":
        classifier = load_classifier()
        print("Legal Sentence Game is ready.")
        print(f"  n-gram corpus : {len(corpus.all_sentences())} sentences")
        print(f"  classifier    : {classifier.backend}"
              + (f" (GPT-2 failed to load: {classifier.load_error})" if classifier.load_error else ""))
        return 0
    if args.command == "play":
        run_game(args.rounds, args.suggestions, args.seed)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
