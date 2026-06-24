# Spooky Author Identification: Classical NLP Baselines, Sparse Features, and Stacking

A compact, end-to-end NLP project on Kaggle's **Spooky Author Identification** competition. The goal is to predict whether a short sentence was written by **Edgar Allan Poe (EAP)**, **Mary Shelley (MWS)**, or **H. P. Lovecraft (HPL)**.

This repository explores a practical question:

> How far can classical NLP go when we choose representations carefully and evaluate them honestly?

The notebook walks through a sequence of classical NLP models: fast linear baselines, richer sparse representations, tuned TF-IDF models, and a stacked ensemble built from out-of-fold predictions. It also includes a representation survey comparing Bag-of-Words, BM25, Word2Vec, and FastText.

## Links

* **Medium article:** [How Far Can Classical NLP Go? From Bag-of-Words to Stacking](https://medium.com/@nahid.nm57/how-far-can-classical-nlp-go-from-bag-of-words-to-stacking-on-spooky-author-identification-a353564427dd)
* **Kaggle competition:** [Spooky Author Identification](https://www.kaggle.com/competitions/spooky-author-identification)
* **Main notebook:** [`main.ipynb`](main.ipynb)


## Project overview

This is a three-class authorship attribution task. Unlike ordinary topic classification, the three authors write in similar Gothic and horror domains, so obvious keywords are not enough. The useful signal is often stylistic: word choice, punctuation, character n-grams, function words, phrase patterns, and sentence-level habits.

The project compares:

* a word-only **Vowpal Wabbit** baseline,
* a richer VW model with punctuation and character n-grams,
* a **TF-IDF** ensemble using word and character features,
* **NB-SVM-style Logistic Regression**,
* **Complement Naive Bayes** and **Multinomial Naive Bayes**,
* **SGD Logistic Regression**,
* a stacked ensemble using out-of-fold predictions and fold averaging,
* appendix-style experiments with Bag-of-Words, BM25 k-NN, Word2Vec, and FastText.

The official Kaggle metric is **multiclass log loss**, so the project emphasizes probability quality as well as accuracy.

## Dataset

The Kaggle data files are **not included** in this repository. Download them from the Kaggle competition page and place them under `data/`:

```text
data/
├── train.csv
└── test.csv
```

Expected dataset sizes:

| Split |   Rows |
| ----- | -----: |
| Train | 19,579 |
| Test  |  8,392 |

Training-label distribution:

| Author | Share |
| ------ | ----: |
| EAP    | 40.3% |
| MWS    | 30.9% |
| HPL    | 28.8% |

## Repository layout

```text
.
├── data/                 # Kaggle data files, ignored by Git
├── outputs/              # Generated VW files, models, predictions, and submissions
├── main.ipynb            # Main project notebook
├── README.md
├── requirements.txt
└── .gitignore
```

`data/` and `outputs/` are intentionally ignored by Git because they contain downloaded data and generated artifacts.

The local files `Medium_Article.md` and `spooky_portfolio_executed.ipynb` are also ignored and are not intended to be committed to GitHub.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project depends mainly on:

* `numpy`
* `pandas`
* `scipy`
* `scikit-learn`
* `gensim`
* `vowpalwabbit`
* `jupyter`

`vowpalwabbit` may require a working compiler or build toolchain on some systems.

## How to run

1. Download `train.csv` and `test.csv` from Kaggle.
2. Place them in the `data/` directory.
3. Start Jupyter from the repository root:

```bash
jupyter notebook main.ipynb
```

4. Run the notebook from top to bottom.

The notebook uses relative paths:

```python
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
```

Generated files, including Vowpal Wabbit data files, saved VW models, prediction files, and the Kaggle submission file, are written to `outputs/`.

## Evaluation protocol

The notebook keeps three result types separate:

1. **70/30 stratified holdout**
   Used for direct comparison across local models.

2. **Full-data level-2 OOF estimate**
   Used as a sanity check for the final stacked meta-learner on the full training set.

3. **Kaggle leaderboard scores**
   Used as the final external test-set evaluation.

The full-data level-2 OOF estimate is not a fully nested cross-validation of the entire pipeline, so it should not be compared directly with the 70/30 holdout rows.

## Metrics

The notebook reports:

* **Accuracy**
* **Macro-F1**
* **Multiclass log loss**

Log loss is the official Kaggle metric. It evaluates the quality of the predicted probability distribution, not only the final predicted class.

## Main results

All holdout rows use the same stratified 70/30 split and are directly comparable.

| Stage                                       |         Evaluation | Accuracy | Macro-F1 |     Log loss |
| ------------------------------------------- | -----------------: | -------: | -------: | -----------: |
| Word-only VW baseline                       |      70/30 holdout |   0.8332 |   0.8323 | not reported |
| Rich VW: words + punctuation + char n-grams |      70/30 holdout |   0.8553 |   0.8552 | not reported |
| Tuned TF-IDF 3-model average                |      70/30 holdout |   0.8601 |   0.8596 |       0.3843 |
| Tuned stacked model with fold averaging     |      70/30 holdout |   0.8687 |   0.8689 |       0.3504 |
| Final stacked model, level-2 OOF estimate   | full training data |   0.8830 |   0.8834 |       0.3167 |

Kaggle leaderboard:

| Submission                | Private log loss | Public log loss |
| ------------------------- | ---------------: | --------------: |
| Final tuned stacked model |          0.30414 |         0.33621 |

## Final stacked model configuration

Best full-data settings for the final stacked submission:

| Component                        | Best setting |
| -------------------------------- | -----------: |
| Logistic Regression              |       `C=10` |
| NB-SVM-style Logistic Regression |       `C=30` |
| Complement Naive Bayes           |  `alpha=0.1` |
| Multinomial Naive Bayes          |  `alpha=0.1` |
| SGD Logistic Regression          | `alpha=3e-6` |
| Meta Logistic Regression         |       `C=30` |

The final Kaggle submission maps the internal class order `[EAP, MWS, HPL]` into Kaggle's required column order:

```text
EAP, HPL, MWS
```

## Representation survey

The notebook also evaluates several foundational representations on the same 70/30 holdout split.

| Representation                     | Accuracy | Macro-F1 | Log loss |
| ---------------------------------- | -------: | -------: | -------: |
| Bag of Words + Logistic Regression |   0.8073 |   0.8058 |   0.5636 |
| BM25 k-NN retriever                |   0.7692 |   0.7680 |   0.6697 |
| Word2Vec + Logistic Regression     |   0.7337 |   0.7328 |   0.6656 |
| FastText + Logistic Regression     |   0.7126 |   0.7111 |   0.7077 |

In this setup, sparse count-based features outperform simple pooled embeddings. That does not mean Word2Vec or FastText are weak in general. It means that for short-text authorship attribution, averaging word vectors loses many of the stylistic details that sparse word, character, and punctuation features preserve.

## Error analysis

The best validated stacked model has relatively balanced recall across authors:

| Author | Correct | Total | Recall |
| ------ | ------: | ----: | -----: |
| EAP    |    2084 |  2370 |  0.879 |
| MWS    |    1558 |  1813 |  0.859 |
| HPL    |    1461 |  1691 |  0.864 |

Most common confusions:

| True -> Predicted | Count |
| ----------------- | ----: |
| MWS -> EAP        |   191 |
| EAP -> MWS        |   166 |
| HPL -> EAP        |   153 |
| EAP -> HPL        |   120 |
| HPL -> MWS        |    77 |
| MWS -> HPL        |    64 |

The errors are not simply a collapse into the majority class. Many difficult cases are short or stylistically neutral sentences where the available evidence is limited.

## Key takeaways

* A fast word-only Vowpal Wabbit model is already a strong baseline.
* Character n-grams and punctuation improve authorship modeling because the task is style-driven.
* TF-IDF word and character features provide a strong sparse representation.
* NB-SVM-style feature reweighting is a simple but effective classical text-classification trick.
* Stacking improves log loss more than raw accuracy, which matters because Kaggle optimizes probability quality.
* Sparse count-based features outperform simple pooled Word2Vec and FastText embeddings in this specific short-text authorship setup.
* Keeping holdout, OOF, and leaderboard results separate prevents overclaiming.

## Limitations and next steps

* Use a fully nested cross-validation setup for a more conservative estimate of the whole modeling pipeline.
* Add explicit calibration diagnostics such as reliability diagrams or expected calibration error.
* Compare against a transformer baseline such as DistilBERT or BERT.
* Run a more systematic hyperparameter search over n-gram ranges, VW settings, smoothing, regularization, and stacking choices.
* Test whether the same conclusions hold on other authorship-attribution datasets.

## Acknowledgements

This project was inspired by the Vowpal Wabbit material from [mlcourse.ai](https://mlcourse.ai) Topic 8, then debugged, extended, and turned into a complete Kaggle submission pipeline.
