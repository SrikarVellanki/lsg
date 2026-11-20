# Legal Sentence Game

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/zpwdhvetvu/badge.svg)](https://ask.thili.ai/b/zpwdhvetvu)
![ci](../../actions/workflows/ci.yml/badge.svg)

A command-line game for law students to practise legal drafting. You are given the start of a
contract sentence (for example *"The Receiving Party shall"*), complete it with the help of an
**n-gram next-word predictor**, and a **fine-tuned GPT-2 model** then tells you which clause type
you actually drafted (Confidentiality, Termination, Indemnification, ...) and how well it matches the
clause you were aiming for.

## What you are building

**Goal.** A CLI game where law students practise drafting legal sentences using n-gram prediction,
receiving instant feedback on the clause type from a fine-tuned GPT-2 model.

**Done when.** Players can complete legal sentences using n-gram predictions, and the completed
sentences are classified by a fine-tuned GPT-2 model, which provides feedback.

## Milestones

- [x] **1. Prepare a legal text corpus for n-gram model training.** `app/corpus.py` (108 labeled clauses across 9 clause types + extra boilerplate) and `clean_text` / `tokenize` / `prepare_corpus` in `app/core.py`.
- [x] **2. Implement the n-gram next-word predictor.** `NGramModel` in `app/core.py`: trigram counts with stupid-backoff, `predict_next()` returns the top-k words with probabilities.
- [x] **3. Load the fine-tuned GPT-2 model for legal clause classification.** `scripts/train_classifier.py` fine-tunes GPT-2; `load_classifier()` loads it for inference.
- [x] **4. Create a CLI game interface presenting partial legal sentences for completion.** `app/cli.py` (`python -m app.cli play`).
- [x] **5. After user completion, classify the full sentence and provide feedback.** `build_feedback()` and `score_result()` in `app/core.py`.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m app.cli hello
pytest
```

### 1. Train the GPT-2 clause classifier (once)

```bash
python -m scripts.train_classifier
```

This downloads the base `gpt2` weights (needs internet, ~500 MB), fine-tunes it on the labeled clauses
in `app/corpus.py`, prints validation accuracy after every epoch, and saves the model to
`models/clause-gpt2/` (git-ignored). Options: `--epochs`, `--batch-size`, `--lr`, `--output`.

> If you skip this step the game still works: it falls back to a simple keyword classifier and tells
> you so at the start. `python -m app.cli hello` shows which classifier is active.

### 2. Play

```bash
python -m app.cli play                       # 3 rounds
python -m app.cli play --rounds 5 --suggestions 4
python -m app.cli play --seed 1              # repeatable prompts
```

Commands inside the game:

| Input | What it does |
|-------|--------------|
| `1`, `2`, `3` | Accept the numbered n-gram suggestion |
| any words | Add your own words to the sentence |
| `undo` | Remove the last thing you added |
| `done` | Finish now and get feedback |
| `skip` / `quit` | Skip the prompt / leave the game |

A sentence also finishes when you end it with a full stop. You score 10 points (+ up to 5 for
classifier confidence) when your sentence is classified as the requested clause type, 2 points for a
sentence that reads as another clause type. Your best total is saved to `DATA_PATH`.

Example:

```
Clause type to draft: Governing Law

  This Agreement shall be governed by ...
    [1] the   (67%)
    [2] and   (33%)
> 1
...
Classifier (gpt2): Governing Law (94% confident)
Target clause type:  Governing Law -> match
Nice drafting - your sentence reads like a Governing Law clause.
Points this round: 15
```

## Project layout

```
app/
  core.py     corpus prep, n-gram model, GPT-2 classifier loader, feedback  (milestones 1,2,3,5)
  cli.py      the interactive game                                        (milestone 4)
  corpus.py   labeled legal clauses and the game prompts
  config.py   reads .env
scripts/train_classifier.py   fine-tunes GPT-2
scripts/evaluate.py           accuracy of GPT-2 vs the keyword baseline on held-out clauses
tests/test_core.py            tests for every milestone (no torch needed)
.github/workflows/ci.yml      runs pytest on every push
```

## Data

The corpus in `app/corpus.py` is **example clause wording written for this teaching project**. It is
modelled on common contract boilerplate, but it is not copied from real contracts or a published
dataset. 108 sentences are labeled with one of 9 clause types (12 each); 6 extra unlabeled boilerplate
sentences are used only by the n-gram model.

## Results

Run `python -m scripts.evaluate --write results.md` after training. It evaluates on the same held-out
validation split that training never sees (18 sentences, 2 per clause type).

| Classifier | Validation accuracy |
|---|---|
| Random guessing (9 labels) | about 11% |
| Keyword baseline (the fallback) | 94% (17/18) |
| Fine-tuned GPT-2 | **TODO: paste your number from `python -m scripts.evaluate`** |

Also paste here: the epoch lines printed by `scripts.train_classifier` and a short example game
session (or a screenshot).

## Baseline and limits

* **Small, clean data.** The validation set is only 18 sentences, so one sentence changes accuracy by
  about 5 points. Treat the numbers as a rough indication, not a benchmark.
* **The keyword baseline is optimistic.** Its keyword lists were written while looking at this same
  corpus, so its 94% is likely higher than it would be on unseen contract text. A fair comparison needs
  clauses from a different source.
* **Only 9 clause types.** Any sentence is forced into one of them, apart from the keyword fallback's
  "Unclear". Both classifiers can be confidently wrong on text that is not a contract clause.
* **Repetitive n-gram suggestions.** With only about 2,400 tokens of training text the n-gram model often
  has only one or two continuations to offer, so suggestions can look alike. A larger corpus or
  Kneser-Ney smoothing would help.
* **Not legal advice.** This is a drafting-practice game; the feedback only says which clause type a
  sentence resembles, not whether it is legally sound.

## Settings and secrets

Settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.**
This project needs no API keys. See `.env.example` for `CLASSIFIER_DIR`, `NGRAM_ORDER` and `DATA_PATH`.

## Limitations

* The corpus is small and written for teaching, so n-gram suggestions are repetitive and the classifier only
  knows 9 clause types. Add more sentences to `app/corpus.py` and retrain to improve both.
* This is a learning tool, not legal advice.

## Stretch ideas

- Kneser-Ney smoothing for better n-gram predictions.
- Scoring over multiple rounds (a simple version is already in: total score and a saved personal best).
- Make the GPT-2 feedback suggest alternative phrasing or highlight common drafting errors.
