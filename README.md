# health-information-robustness-nlp
NLP classification and robustness analysis for dataset-defined health-information reliability and human-vs-AI originality.
# Health Information NLP: Performance, Robustness and Shortcut Learning

NLP classification and robustness analysis using the MM-Health dataset, developed as part of an MSc Data Science and Artificial Intelligence dissertation at Queen Mary University of London.

This project evaluates whether strong benchmark performance in health-information classification remains reliable when the data distribution changes. Rather than focusing only on headline test scores, the analysis investigates sensitivity to authorship, source, generator and dataset-specific structural or lexical cues.

## Project Overview

The project examines two dataset-defined text-classification tasks:

- **Reliability classification:** distinguishing reliable from unreliable human-written health articles.
- **Originality classification:** distinguishing human-written from AI-generated articles.

Reliability labels are inherited from the component datasets used to construct MM-Health and should therefore be interpreted as a dataset-defined classification target rather than direct medical fact checking.

Similarly, originality classification refers to the human and AI-generated conditions represented within MM-Health and should not be interpreted as universal AI-text detection.

## Models

Five model configurations were evaluated:

- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- BERT
- RoBERTa
- PubMedBERT

RoBERTa achieved the strongest performance on the standard benchmark and was therefore selected for the main robustness experiments.

## Key Results

| Experiment | Macro-F1 |
|---|---:|
| Standard RoBERTa — reliability | 0.952 |
| Standard RoBERTa — originality | 1.000 |
| Reliability on matched AI-generated versions | 0.614 |
| Reliability on mixed human–AI test set | 0.761 |
| Cross-source reliability | 0.785–0.941 |
| Held-out Mistral originality | 0.591 ± 0.092 |
| Held-out Mistral after leading-Markdown normalisation | 0.984 ± 0.015 |
| Length-only originality baseline | 0.631 |
| Style-only originality classifier | 0.875 |
| Normalised TF-IDF lexical classifier | 0.988 |
| Normalised RoBERTa originality | 0.999 |

The main finding is that near-perfect performance on a familiar benchmark does not necessarily imply robust generalisation.

For reliability classification, performance fell substantially when the same underlying test articles were represented as AI-generated text. Cross-source testing also produced lower performance when individual MM-Health component datasets were excluded from training.

For originality classification, a generator-specific leading-Markdown formatting pattern contributed substantially to poor transfer to held-out Mistral. Removing this cue increased mean macro-F1 from 0.591 to 0.984. Further experiments showed that structural, stylistic and lexical differences remained strongly predictive within MM-Health.

## Robustness and Shortcut Analysis

The project includes:

- matched human–AI authorship-shift testing
- mixed-authorship evaluation
- cross-source reliability evaluation
- cross-generator originality evaluation
- paired prediction-level error analysis
- leading-Markdown normalisation
- length-only classification
- paired stylometric analysis
- style-only logistic regression
- TF-IDF lexical analysis

Statistical analysis included McNemar testing with Holm correction, Wilcoxon signed-rank testing with Benjamini–Hochberg correction, and paired effect-size analysis.

## Dataset

The project uses the textual component of **MM-Health**, which combines human-written health articles from four existing datasets with AI-generated versions produced by five language models.

The official article-grouped train, validation and test partitions were retained to prevent versions of the same underlying article appearing across different splits.

The dataset itself is not redistributed in this repository.

## Tools and Libraries

- Python
- pandas
- NumPy
- SciPy
- scikit-learn
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Matplotlib
- Jupyter / Google Colab

## Repository Structure

```text
health-information-robustness-nlp/
├── notebooks/
│   ├── PG_Project_Full_Pipeline_AdelindSho.ipynb
│   └── README.md
├── .gitignore
└── README.md
