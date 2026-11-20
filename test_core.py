"""Tests for the Legal Sentence Game.

They do not need torch/transformers: the classifier falls back to keywords when no fine-tuned
GPT-2 folder exists, so these also run on GitHub Actions.
"""

import os
import tempfile

from app import corpus
from app.cli import build_parser, load_best_score, play_round, run_game, save_best_score
from app.core import (
    ClauseClassifier,
    ClauseResult,
    NGramModel,
    build_feedback,
    build_ngram_model,
    clean_text,
    join_text,
    load_classifier,
    prepare_corpus,
    score_result,
    sentence_is_complete,
    split_sentences,
    tokenize,
)

NO_MODEL_DIR = os.path.join(tempfile.gettempdir(), "definitely-not-a-model-dir")


def scripted(*answers):
    """Return an input() replacement that replays ``answers``."""
    it = iter(answers)
    return lambda _prompt="": next(it)


# ------------------------------------------------------------------ plumbing
def test_cli_parser_knows_commands():
    assert build_parser().parse_args(["hello"]).command == "hello"
    args = build_parser().parse_args(["play", "--rounds", "2"])
    assert args.command == "play" and args.rounds == 2


def test_corpus_labels_are_consistent():
    labels = {label for label, _ in corpus.LABELED_CLAUSES}
    assert labels == set(corpus.LABELS)
    assert all(p.clause_type in corpus.LABELS for p in corpus.GAME_PROMPTS)


# ------------------------------------------------------------------ milestone 1
def test_clean_text_normalises_quotes_and_whitespace():
    assert clean_text("  The\u2019s   \n clause ") == "The's clause"


def test_tokenize_lowercases_and_keeps_punctuation():
    assert tokenize("The Party shall pay, promptly.") == ["the", "party", "shall", "pay", ",", "promptly", "."]


def test_split_sentences():
    assert len(split_sentences("One thing. Another thing. A third.")) == 3


def test_prepare_corpus_default_and_custom():
    assert len(prepare_corpus()) == len(corpus.all_sentences())
    assert prepare_corpus("Hi. The Tenant shall pay rent monthly.") == [
        ["the", "tenant", "shall", "pay", "rent", "monthly", "."]
    ]


# ------------------------------------------------------------------ milestone 2
def test_ngram_predicts_seen_continuation():
    model = NGramModel(3).fit([tokenize("The Tenant shall pay rent."), tokenize("The Tenant shall pay tax.")])
    words = [w for w, _ in model.predict_next(tokenize("the tenant shall"), k=1)]
    assert words == ["pay"]


def test_ngram_probabilities_are_sensible():
    model = NGramModel(2).fit([tokenize("a b c ."), tokenize("a b d .")])
    assert abs(model.probability("b", ["a"]) - 1.0) < 1e-9
    assert abs(model.probability("c", ["a", "b"]) - 0.5) < 1e-9


def test_ngram_backs_off_for_unseen_context():
    model = build_ngram_model(3)
    suggestions = model.predict_next(tokenize("zebra quantum"), k=3)
    assert len(suggestions) == 3  # still gives suggestions instead of crashing


def test_ngram_is_deterministic_and_sorted():
    model = build_ngram_model(3)
    a = model.predict_next(tokenize("This Agreement shall be governed by"), k=5)
    assert a == model.predict_next(tokenize("This Agreement shall be governed by"), k=5)
    assert [p for _, p in a] == sorted((p for _, p in a), reverse=True)
    assert a[0][0] == "the"


def test_ngram_never_suggests_special_tokens():
    model = build_ngram_model(3)
    words = [w for w, _ in model.predict_next(tokenize("shall"), k=50)]
    assert "<s>" not in words and "</s>" not in words


# ------------------------------------------------------------------ milestone 3
def test_classifier_falls_back_to_keywords_when_no_model():
    clf = load_classifier(NO_MODEL_DIR)
    assert isinstance(clf, ClauseClassifier) and clf.backend == "keyword"


def test_keyword_classifier_is_mostly_right_on_corpus():
    clf = ClauseClassifier(NO_MODEL_DIR)
    hits = sum(clf.classify(text).label == label for label, text in corpus.LABELED_CLAUSES)
    assert hits / len(corpus.LABELED_CLAUSES) >= 0.85


def test_classifier_handles_empty_and_gibberish():
    clf = ClauseClassifier(NO_MODEL_DIR)
    assert clf.classify("").label == "Unclear"
    assert clf.classify("zzz qqq").label == "Unclear"


def test_classifier_result_has_scores():
    result = ClauseClassifier(NO_MODEL_DIR).classify("This Agreement shall be governed by the laws of India.")
    assert result.label == "Governing Law" and 0 < result.confidence <= 1
    assert result.top(1)[0][0] == "Governing Law"


# ------------------------------------------------------------------ milestone 5
def test_score_result():
    good = ClauseResult("Payment", 1.0, "keyword", {"Payment": 1.0})
    wrong = ClauseResult("Warranty", 1.0, "keyword", {"Warranty": 1.0})
    unclear = ClauseResult("Unclear", 0.0, "keyword", {})
    assert score_result("Payment", good) == 15
    assert score_result("Payment", wrong) == 2
    assert score_result("Payment", unclear) == 0


def test_feedback_for_match_and_mismatch():
    match = ClauseResult("Payment", 0.9, "gpt2", {"Payment": 0.9, "Warranty": 0.1})
    text = build_feedback("Payment", "The Client shall pay all invoices within thirty days.", match)
    assert "match" in text and "Payment" in text and "90%" in text

    miss = ClauseResult("Warranty", 0.8, "gpt2", {"Warranty": 0.8, "Payment": 0.2})
    text = build_feedback("Payment", "The Supplier warrants quality", miss)
    assert "mismatch" in text and "more like Warranty" in text and "full stop" in text


# ------------------------------------------------------------------ milestone 4 (CLI)
def test_join_text_spacing():
    assert join_text("The Party", "shall") == "The Party shall"
    assert join_text("The Party shall pay", ".") == "The Party shall pay."
    assert sentence_is_complete("Done.") and not sentence_is_complete("Not yet")


def test_play_round_full_flow():
    model = build_ngram_model(3)
    clf = ClauseClassifier(NO_MODEL_DIR)
    out = []
    prompt = corpus.GamePrompt("Governing Law", "This Agreement shall be governed by")
    sentence, points = play_round(
        prompt, model, clf, input_fn=scripted("1", "laws of India ."), print_fn=out.append
    )
    assert sentence == "This Agreement shall be governed by the laws of India."
    assert points >= 10  # classified as the intended clause type
    assert any("Governing Law" in line for line in out)


def test_play_round_undo_skip_and_help():
    model = build_ngram_model(3)
    clf = ClauseClassifier(NO_MODEL_DIR)
    out = []
    prompt = corpus.GamePrompt("Payment", "The Client shall pay")
    assert play_round(prompt, model, clf, input_fn=scripted("help", "undo", "1", "undo", "skip"), print_fn=out.append) is None
    assert any("How to play" in line for line in out)


def test_play_round_done_command():
    out = []
    prompt = corpus.GamePrompt("Payment", "The Client shall pay")
    sentence, _ = play_round(
        prompt, build_ngram_model(3), ClauseClassifier(NO_MODEL_DIR), input_fn=scripted("the invoice", "done"), print_fn=out.append
    )
    assert sentence == "The Client shall pay the invoice"


def test_run_game_scores_and_saves_best():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "app.json")
        out = []
        total = run_game(
            rounds=1, seed=0, input_fn=scripted("done"), print_fn=out.append,
            classifier=ClauseClassifier(NO_MODEL_DIR), data_path=path,
        )
        assert total >= 0
        assert any("Final score" in line for line in out)
        save_best_score(path, 99)
        assert load_best_score(path) == 99


def test_run_game_quit_is_graceful():
    out = []
    run_game(rounds=2, seed=0, input_fn=scripted("quit"), print_fn=out.append,
             classifier=ClauseClassifier(NO_MODEL_DIR), data_path=os.path.join(tempfile.gettempdir(), "lsg-quit.json"))
    assert any("Leaving the game early" in line for line in out)


# ------------------------------------------------------------------ evaluation script
def test_evaluate_keyword_baseline_on_validation_split():
    from scripts.evaluate import evaluate, format_report
    from scripts.train_classifier import stratified_split

    train, val = stratified_split(corpus.LABELED_CLAUSES, 0.2, 42)
    assert not set(train) & set(val)  # no sentence is in both sets
    result = evaluate(ClauseClassifier(NO_MODEL_DIR), val)
    assert result["n"] == len(val) and result["accuracy"] > 1 / len(corpus.LABELS)
    assert "Keyword baseline" in format_report({"Keyword baseline": result}, len(train), len(val))
