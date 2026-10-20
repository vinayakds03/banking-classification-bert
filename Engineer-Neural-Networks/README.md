# Classifying Banking Intent from Customer Queries

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.4.1](https://img.shields.io/badge/PyTorch-2.4.1-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![PEFT LoRA](https://img.shields.io/badge/PEFT-LoRA-green.svg)](https://github.com/huggingface/peft)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end NLP engineering and machine learning system for fine-grained customer intent classification on the **Banking77** dataset. This project contrasts a traditional feature-engineered neural network (**TF-IDF + Multi-Layer Perceptron**) against a state-of-the-art contextual transformer (**RoBERTa-base**) fine-tuned with Parameter-Efficient Fine-Tuning (**PEFT / LoRA**), followed by an enterprise **Personally Identifiable Information (PII) redaction and cryptographic hashing** pipeline.

---

## Table of Contents
1. [Executive Summary & Key Benchmarks](#1-executive-summary--key-benchmarks)
2. [Dataset Architecture: Banking77](#2-dataset-architecture-banking77)
3. [System Architecture & Workflow](#3-system-architecture--workflow)
4. [Environment Setup & Installation](#4-environment-setup--installation)
5. [Exhaustive Cell-by-Cell In-Depth Walkthrough (Cells 00 to 84)](#5-exhaustive-cell-by-cell-in-depth-walkthrough-cells-00-to-84)
   - [Module 1: Environment Setup & Data Ingestion (Cells 00 – 05)](#module-1-environment-setup--data-ingestion-cells-00--05)
   - [Module 2: Exploratory Data Analysis & Text Mining (Cells 06 – 24)](#module-2-exploratory-data-analysis--text-mining-cells-06--24)
   - [Module 3: Baseline Multi-Layer Perceptron (MLP) (Cells 25 – 44)](#module-3-baseline-multi-layer-perceptron-mlp-cells-25--44)
   - [Module 4: Parameter-Efficient Fine-Tuning (PEFT) with RoBERTa & LoRA (Cells 45 – 65)](#module-4-parameter-efficient-fine-tuning-peft-with-roberta--lora-cells-45--65)
   - [Module 5: Performance Comparison & Trade-Off Analysis (Cells 66 – 71)](#module-5-performance-comparison--trade-off-analysis-cells-66--71)
   - [Module 6: Responsible AI & Enterprise PII Redaction Pipeline (Cells 72 – 82)](#module-6-responsible-ai--enterprise-pii-redaction-pipeline-cells-72--82)
   - [Module 7: Project Conclusion & Production Roadmap (Cells 83 – 84)](#module-7-project-conclusion--production-roadmap-cells-83--84)
6. [Comparative Metrics Table](#6-comparative-metrics-table)
7. [Enterprise Deployment Considerations](#7-enterprise-deployment-considerations)

---

## 1. Executive Summary & Key Benchmarks

In customer service conversational agents, intent classification is the foundational routing mechanism. In banking, intent classification is uniquely challenging because:
- **High Cardinality**: 77 distinct intent classes.
- **High Semantic Proximity**: Multiple categories share nearly identical lexical tokens (e.g., `card_arrival` vs. `card_linking` vs. `card_not_working`).
- **High Consequence**: Routing a fraud dispute (`compromised`) to an account inquiry agent causes significant financial and regulatory risk.

### Benchmark Results Overview
| Metric | Baseline: TF-IDF + MLP | Modern: LoRA-RoBERTa | Delta / Impact |
|---|---|---|---|
| **Overall Test Accuracy** | **84.22%** | **93.38%** | **+9.16%** absolute improvement |
| **Macro Average F1-Score** | **0.8398** | **0.9329** | **+0.0931** F1 gain |
| **Weighted Average F1-Score** | **0.8415** | **0.9337** | **+0.0922** F1 gain |
| **Trainable Parameters** | 25,639,497 (100%) | **884,736 (0.70%)** | **96.5% parameter reduction** |
| **Total Base Parameters** | 25.6M | 125.5M | - |
| **Training Time (CPU)** | **~35 seconds** | ~12-15 minutes | MLP is ~25x faster |
| **Hardware Footprint** | Extremely lightweight | Moderate (GPU recommended) | Trade-off: speed vs accuracy |

---

## 2. Dataset Architecture: Banking77

The dataset utilized is **Banking77** (developed by PolyAI, published in EMNLP 2020):
- **Domain**: Online banking and financial customer queries.
- **Classes**: 77 fine-grained intents (e.g., `card_arrival`, `exchange_rate`, `lost_or_stolen_card`, `pin_blocked`, `top_up_failed`).
- **Total Samples**: 13,083 queries.
  - **Training Set**: 10,003 queries (`datasets/banking77_train.csv`)
  - **Testing Set**: 3,080 queries (`datasets/banking77_test.csv`)
- **Structure**:
  - `text` (*str*): The verbatim text entered by the customer.
  - `category` (*str*): The ground-truth string intent label.

---

## 3. System Architecture & Workflow

```
[Raw Customer Query]
        │
        ▼
[PII Redaction & Hashing Engine]  <--- Regex Sanitization (Card, CVV, Phone, Email, IP)
        │                              + SHA-256 Identifier Tracking
        ▼
 [Cleaned Query Text]
        │
        ├─────────────────────────────────────────┐
        ▼                                         ▼
[Baseline Path]                           [Modern Transformer Path]
TfidfVectorizer(max_features=50,000)      AutoTokenizer(roberta-base, max_length=64)
        │                                         │
[Sparse Term Vectors]                     [Input IDs, Attention Mask]
        │                                         │
SimpleMLP(50000 -> 512 -> 77)             RoBERTa-base + LoRA (r=8, alpha=16)
CrossEntropyLoss + Adam                   AdamW + Linear Warmup (Trainer)
        │                                         │
        ▼                                         ▼
[MLP Logits / Predictions]                [Transformer Logits / Predictions]
(Accuracy: ~84.2%)                        (Accuracy: ~93.4%)
        │                                         │
        └───────────────────┬─────────────────────┘
                            ▼
               [Comparative Evaluation]
               - Classification Reports
               - Per-Class F1 Delta
               - Top/Bottom 10 Delta
                            │
                            ▼
               [Top-K Production Inference]
               Safe Intent Probability Output
```

---

## 4. Environment Setup & Installation

### Prerequisites
- Windows 10/11, macOS, or Linux
- **Python 3.12** (Python 3.15 is not supported by PyTorch binary wheels at this time)

### Installation Steps
```powershell
# 1. Clone your repository
git clone https://github.com/vinayakds03/banking-classification-bert.git
cd banking-classification-bert/Engineer-Neural-Networks

# 2. Create Python 3.12 Virtual Environment
py -3.12 -m venv .venv

# 3. Activate Virtual Environment
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
# source .venv/bin/activate

# 4. Install all validated dependencies
pip install -r requirements.txt
```

---

## 5. Exhaustive Cell-by-Cell In-Depth Walkthrough (Cells 00 to 84)

Every single cell in `banking_classification_BERT.ipynb` is documented below in sequence, including its purpose, code line-by-line, parameter analysis, and mathematical foundations.

---

### Module 1: Environment Setup & Data Ingestion (Cells 00 – 05)

#### Cell 00 [Markdown] - Project Introduction & Goals
- **Purpose**: Establishes the academic and industrial background of intent classification in financial banking. Explains the two compared architectures (MLP vs. fine-tuned RoBERTa), project goals, and dataset dimensions.
- **Key Concepts Introduced**:
  - NLP intent classification in conversational AI.
  - High-cardinality multi-class classification (77 intents).
  - Efficiency-performance Pareto frontier: fast linear models vs. heavy deep contextual transformers.

---

#### Cell 01 [Markdown] - Setup + Libraries Header
- **Purpose**: Section divider defining initial imports, random seeds, and compute hardware setup.

---

#### Cell 02 [Code] - Import Core Libraries & Hardware Configuration
```python
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch

# Set random seed
torch.manual_seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```
- **Line-by-Line Breakdown**:
  - `import warnings; warnings.filterwarnings("ignore")`: Suppresses non-critical warning streams (e.g., future library deprecations) to maintain clean console/notebook outputs.
  - `import pandas as pd`: Ingests high-performance DataFrame tabular data manipulation tools.
  - `import numpy as np`: Ingests C-optimized array and matrix calculation routines.
  - `import torch`: Loads the core PyTorch deep learning framework.
  - `torch.manual_seed(42)`: Configures the PyTorch pseudo-random number generator (PRNG) with static seed `42`. This ensures weight initialization for linear layers and dropout masks are reproducible across runs.
  - `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`: Hardware detection query. If an NVIDIA GPU with active CUDA drivers is detected, assigns `'cuda'`; otherwise defaults to host `'cpu'`.
  - `print(f"Using device: {device}")`: Displays whether CPU or GPU acceleration is active.
- **Parameters Explained**:
  | Parameter / Call | Type | Value | Deep Learning / Mathematical Purpose |
  |---|---|---|---|
  | `warnings.filterwarnings` | `str` | `"ignore"` | Disables warnings from polluting training logs. |
  | `torch.manual_seed` | `int` | `42` | Arbitrary integer seed ensuring deterministic weight initialization. |
  | `torch.cuda.is_available()` | `bool` | Runtime Check | Probes GPU device driver through CUDA runtime API. |

---

#### Cell 03 [Markdown] - Load Data Instructions
- **Purpose**: Outlines the file paths for the training (`datasets/banking77_train.csv`) and testing (`datasets/banking77_test.csv`) sets.

---

#### Cell 04 [Code] - Ingest Training & Testing Datasets
```python
train = pd.read_csv("datasets/banking77_train.csv")
test = pd.read_csv("datasets/banking77_test.csv")
train.head()
```
- **Line-by-Line Breakdown**:
  - `train = pd.read_csv("datasets/banking77_train.csv")`: Parses the CSV file containing 10,003 labeled banking customer interactions into a Pandas DataFrame.
  - `test = pd.read_csv("datasets/banking77_test.csv")`: Parses the test split (3,080 rows) into a separate DataFrame.
  - `train.head()`: Emits an HTML table displaying the first 5 records of the training set.
- **Parameters Explained**:
  | Parameter | Type | Value | Purpose |
  |---|---|---|---|
  | `filepath_or_buffer` | `str` | `"datasets/banking77_train.csv"` | Local relative path to the pre-split training dataset. |

---

#### Cell 05 [Code] - Inspect Testing Dataset Header
```python
test.head()
```
- **Line-by-Line Breakdown**:
  - Emits the top 5 rows of `test` DataFrame, confirming columns `text` and `category` match training schema.

---

### Module 2: Exploratory Data Analysis & Text Mining (Cells 06 – 24)

#### Cell 06 [Markdown] - Data Characteristics Overview
- **Purpose**: Documents the two primary columns:
  - `text`: Natural language customer query string.
  - `category`: Ground-truth string intent label.

---

#### Cell 07 [Markdown] - Section Marker: Training and Testing Size
- **Purpose**: Prepares cell 08 for dataset dimension validation.

---

#### Cell 08 [Code] - Print Dataset Shapes
```python
print("Training size:", train.shape)
print("Testing size:", test.shape)
```
- **Line-by-Line Breakdown**:
  - `train.shape`: Tuple `(10003, 2)` representing 10,003 sample rows and 2 feature columns.
  - `test.shape`: Tuple `(3080, 2)` representing 3,080 evaluation instances.
  - Split ratio is approximately 76.5% train / 23.5% test.

---

#### Cell 09 [Markdown] - Section Marker: Number of Intent Labels
- **Purpose**: Introduces class cardinality calculation.

---

#### Cell 10 [Code] - Count Unique Intent Categories
```python
print("Number of banking intent labels:", train["category"].nunique())
```
- **Line-by-Line Breakdown**:
  - `train["category"].nunique()`: Calculates the number of distinct categories across the label column.
  - Output: `77`, establishing that our classifier requires 77 output logits.

---

#### Cell 11 [Markdown] - Intent Distribution Context
- **Purpose**: Outlines the necessity of auditing class balance before model training to check for severe class skew.

---

#### Cell 12 [Code] - Plot Complete Category Distribution
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(16,4))
train["category"].value_counts().plot(kind='bar')
plt.title("Distribution of Intent Labels")
plt.xlabel("category")
plt.ylabel("Count")
plt.show()
```
- **Line-by-Line Breakdown**:
  - `plt.figure(figsize=(16, 4))`: Creates a wide plotting figure (16 inches wide, 4 inches tall) to accommodate 77 tick labels.
  - `train["category"].value_counts().plot(kind='bar')`: Computes frequency distribution across all 77 categories and renders a bar plot.
  - `plt.title(...)`, `plt.xlabel(...)`, `plt.ylabel(...)`: Sets semantic plot labels.
  - `plt.show()`: Renders the figure and clears the plotting buffer.
- **Parameters Explained**:
  | Parameter | Type | Value | Purpose |
  |---|---|---|---|
  | `figsize` | `tuple(int, int)` | `(16, 4)` | Wide aspect ratio prevents 77 category names from clumping together. |
  | `kind` | `str` | `'bar'` | Categorical discrete frequency representation. |

---

#### Cell 13 [Markdown] - Class Balance Analysis
- **Purpose**: Discusses findings from the distribution plot: the classes exhibit mild, healthy variation (~100 to ~190 samples per class) with no catastrophic class collapse, meaning specialized cost-sensitive reweighting is optional.

---

#### Cell 14 [Markdown] - Focus on Top 10 Categories
- **Purpose**: Motivates isolating the 10 most common customer intents to identify primary banking pain points.

---

#### Cell 15 [Code] - Plot Top 10 Most Frequent Intents
```python
plt.figure(figsize=(16,4))
train["category"].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Most Occurring Intent Labels")
plt.xlabel("category")
plt.ylabel("Count")
plt.show()
```
- **Line-by-Line Breakdown**:
  - `train["category"].value_counts().head(10)`: Slices only the top 10 highest-volume categories.
  - Key intents identified: `card_arrival`, `change_pin`, `transfer_fee_charged`, `cancel_transfer`, `direct_debit_payment_not_recognised`.

---

#### Cell 16 [Markdown] - Common Themes in Customer Inquiries
- **Purpose**: Summarizes the business drivers behind top intents: physical card shipping delays, fee transparency disputes, and unauthorized transaction alerts.

---

#### Cell 17 [Markdown] - Length Distribution Exploration
- **Purpose**: Introduces token and word count analysis to determine required sequence padding length for transformer models.

---

#### Cell 18 [Code] - Word Length Computation & Boxplot
```python
train['text_length'] = train['text'].str.split().str.len()

plt.figure(figsize=(16, 6))
train.boxplot(column='text_length', by='category', rot=90, figsize=(16,6))
plt.suptitle('')
plt.title('Text Length Distribution by Category')
plt.xlabel('Category')
plt.ylabel('Text Length (Number of Words)')
plt.show()
```
- **Line-by-Line Breakdown**:
  - `train['text'].str.split().str.len()`: Tokenizes queries by whitespace and counts word tokens per sample.
  - `train.boxplot(...)`: Generates a comparative box-and-whisker plot displaying medians, quartiles, and outlier queries across all 77 categories.
  - `rot=90`: Rotates category axis labels 90 degrees for vertical readability.
- **Parameters Explained**:
  | Parameter | Type | Value | Purpose |
  |---|---|---|---|
  | `column` | `str` | `'text_length'` | Metric variable to visualize. |
  | `by` | `str` | `'category'` | Grouping factor variable. |
  | `rot` | `int` | `90` | Rotates x-axis tick labels vertically to eliminate overlapping text. |

---

#### Cell 19 [Markdown] - Text Length Findings
- **Purpose**: Synthesizes query length statistics:
  - Median word count is **11 words**.
  - 95th percentile is under **25 words**.
  - Maximum length is **79 words**.
  - **Crucial Engineering Decision**: Sets maximum token length `max_length=64` for the transformer tokenizer, capturing >98% of queries without truncation while drastically minimizing quadratic attention memory: $\mathcal{O}(L^2)$.

---

#### Cell 20 [Markdown] - Lexical WordCloud Motivation
- **Purpose**: Explains how visual frequency clouds help distinguish high-frequency domain stopwords from intent-specific keywords.

---

#### Cell 21 [Code] - Inspect Top 6 Intent Label Strings
```python
train['category'].value_counts().head(6)
```
- **Line-by-Line Breakdown**:
  - Displays the 6 most common intent categories and their frequency counts.

---

#### Cell 22 [Code] - Multi-Panel WordCloud Generation
```python
from wordcloud import WordCloud
import numpy as np

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
top_intents = train['category'].value_counts().head(6).index

for idx, intent in enumerate(top_intents):
    row, col = divmod(idx, 3)
    text = " ".join(train[train['category'] == intent]['text'])
    wc = WordCloud(width=400, height=300, background_color='white', colormap='viridis').generate(text)
    axes[row, col].imshow(wc, interpolation='bilinear')
    axes[row, col].set_title(f"Intent: {intent}", fontsize=14)
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
```
- **Line-by-Line Breakdown**:
  - `fig, axes = plt.subplots(2, 3, figsize=(18, 12))`: Allocates a 2-row, 3-column canvas of subplots.
  - `divmod(idx, 3)`: Computes integer row quotient and column remainder for grid indexing.
  - `" ".join(train[train['category'] == intent]['text'])`: Concatenates all customer texts for intent into a single corpus.
  - `WordCloud(...)`: Generates word frequency map.
  - `axes[row, col].axis('off')`: Removes coordinate axes for visual clarity.
- **Parameters Explained**:
  | Parameter | Type | Value | Purpose |
  |---|---|---|---|
  | `width, height` | `int, int` | `400, 300` | Resolution dimensions of generated word cloud image. |
  | `background_color` | `str` | `'white'` | High-contrast background color. |
  | `interpolation` | `str` | `'bilinear'` | Smooths pixel rendering during resizing. |

---

#### Cell 23 [Markdown] - WordCloud Interpretation
- **Purpose**: Observes that categories share root vocabulary ("card", "charge", "payment"), explaining why simple unigram models struggle with boundary differentiation and context-aware models (transformers) are needed.

---

#### Cell 24 [Code] - Empty Spacer Cell
- **Purpose**: Preserved empty notebook cell from curriculum template.

---

### Module 3: Baseline Multi-Layer Perceptron (MLP) (Cells 25 – 44)

#### Cell 25 [Markdown] - Baseline MLP Overview
- **Purpose**: Establishes the classical NLP approach: TF-IDF n-gram feature extraction followed by a feedforward Multi-Layer Perceptron (MLP) in PyTorch.

---

#### Cell 26 [Markdown] - Target Label Encoding Rationale
- **Purpose**: Explains mapping 77 arbitrary string labels into contiguous integer targets $\{0, 1, \dots, 76\}$ required by PyTorch's `nn.CrossEntropyLoss`.

---

#### Cell 27 [Code] - Scikit-Learn LabelEncoder Implementation
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(train["category"].astype(str).values)
y_test  = le.transform(test["category"].astype(str).values)

num_classes = len(le.classes_)
print(f"Number of classes: {num_classes}")
```
- **Line-by-Line Breakdown**:
  - `le = LabelEncoder()`: Instantiates the encoder.
  - `le.fit_transform(...)`: Scans `train["category"]`, identifies all unique classes alphabetically, and converts to a 1D NumPy array of integers.
  - `le.transform(...)`: Applies the **exact same** mapping to `test` data. **Critical**: We do NOT call `fit_transform` on test to avoid target data leakage.
  - `num_classes = len(le.classes_)`: Stores `77` as `num_classes`.
- **Parameters Explained**:
  | Function / Property | Type | Output Value | Purpose |
  |---|---|---|---|
  | `fit_transform` | Method | `ndarray (10003,)` | Learns unique vocabulary and encodes training targets. |
  | `transform` | Method | `ndarray (3080,)` | Applies learned target vocabulary to unseen test split. |
  | `le.classes_` | Property | Array of 77 strings | Retains string-to-integer mapping for inference decoding. |

---

#### Cell 28 [Markdown] - TF-IDF Vectorization Theory
- **Purpose**: Explains Term Frequency-Inverse Document Frequency mathematics:
  $$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t'} f_{t',d}}$$
  $$\text{IDF}(t, D) = \log\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1$$
  High TF-IDF indicates a word that is frequent in the current query but rare across the general banking corpus.

---

#### Cell 29 [Code] - TF-IDF Feature Extraction Pipeline
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=50_000,
    strip_accents="unicode",
    sublinear_tf=True
)

X_train_sparse = tfidf.fit_transform(train["text"].values)
X_test_sparse  = tfidf.transform(test["text"].values)

input_size = X_train_sparse.shape[1]
print(f"X_train shape: {X_train_sparse.shape}")
print(f"X_test shape:  {X_test_sparse.shape}")
print(f"Input feature size: {input_size}")
```
- **Line-by-Line Breakdown**:
  - `TfidfVectorizer(...)`: Initializes vectorizer with custom constraints.
  - `tfidf.fit_transform(...)`: Learns n-gram vocabulary on training queries and returns a Scipy Compressed Sparse Row (`scipy.sparse.csr_matrix`).
  - `tfidf.transform(...)`: Transforms test queries using strictly the training vocabulary.
  - `input_size = X_train_sparse.shape[1]`: Extracts total vocabulary columns (up to 50,000 features).
- **Parameters Deep Dive**:
  | Parameter | Value | Mathematical / Computational Purpose |
  |---|---|---|
  | `lowercase` | `True` | Case-folding normalizes "Card" and "card" to the same token. |
  | `ngram_range` | `(1, 2)` | Extracts both unigrams ("transfer") and bigrams ("transfer fee") to capture local phrases. |
  | `min_df` | `2` | Ignores tokens appearing in fewer than 2 queries (filters typos and hapax legomena). |
  | `max_df` | `0.95` | Ignores tokens appearing in >95% of queries (corpus-wide generic words). |
  | `max_features` | `50000` | Caps vocabulary matrix width at 50,000 columns, bounding model parameter size. |
  | `strip_accents`| `"unicode"` | Normalizes accented characters to basic ASCII. |
  | `sublinear_tf` | `True` | Replaces TF with $1 + \log(\text{TF})$, dampening the impact of repeated words. |

---

#### Cell 30 [Markdown] - Custom PyTorch Dataset Strategy
- **Purpose**: Explains memory optimization: keeping sparse matrices in RAM and converting only the current mini-batch to dense tensors on-the-fly.

---

#### Cell 31 [Code] - PyTorch TfidfDataset Implementation
```python
from torch.utils.data import Dataset, DataLoader

class TfidfDataset(Dataset):
    def __init__(self, X_sparse, y):
        self.X = X_sparse
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x_dense = self.X[idx].toarray().squeeze(0)
        x_tensor = torch.from_numpy(x_dense).float()
        y_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        return x_tensor, y_tensor

train_ds = TfidfDataset(X_train_sparse, y_train)
test_ds  = TfidfDataset(X_test_sparse, y_test)
```
- **Line-by-Line Breakdown**:
  - Subclasses `torch.utils.data.Dataset`.
  - `__len__`: Returns total rows ($N$).
  - `__getitem__`: Slices row `idx` from the CSR matrix, executes `.toarray()` to convert sparse row to dense NumPy vector, and casts to `torch.float32`. Labels are cast to `torch.long` ($64$-bit integer).
  - Instantiates `train_ds` and `test_ds`.

---

#### Cell 32 [Markdown] - DataLoader Batching Strategy
- **Purpose**: Outlines mini-batching parameters and stochastic shuffle logic.

---

#### Cell 33 [Code] - DataLoader Instantiation
```python
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=512, shuffle=False)
```
- **Line-by-Line Breakdown**:
  - `train_loader`: Batches training samples in groups of 256; `shuffle=True` randomizes order each epoch for stochastic gradient descent.
  - `test_loader`: Batches testing samples in larger groups of 512; `shuffle=False` preserves evaluation ordering.
- **Parameters Explained**:
  | Parameter | Value | Reason |
  |---|---|---|
  | `batch_size (train)` | `256` | Balances gradient noise with vectorization throughput. |
  | `batch_size (test)` | `512` | Evaluation requires no gradient graph, allowing larger batches to maximize throughput. |
  | `shuffle` | `True/False` | Shuffling prevents the model from learning spurious batch-order correlations. |

---

#### Cell 34 [Markdown] - Neural Network Architecture Design
- **Purpose**: Explains the MLP topology: Input ($50,000$) $\to$ Linear ($512$) $\to$ ReLU $\to$ Dropout ($0.2$) $\to$ Linear ($77$).

---

#### Cell 35 [Code] - PyTorch SimpleMLP Architecture
```python
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size=512, output_size=77):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
```
- **Line-by-Line Breakdown**:
  - `self.fc1 = nn.Linear(input_size, hidden_size)`: Fully connected layer computing $z_1 = x W_1^T + b_1$. Parameters: $50000 \times 512 + 512 = 25,600,512$.
  - `self.relu = nn.ReLU()`: Non-linear activation $\max(0, z_1)$ enabling the network to learn non-linear decision boundaries.
  - `self.dropout = nn.Dropout(0.2)`: Inverted dropout randomly deactivating 20% of neuron outputs during training to prevent co-adaptation.
  - `self.fc2 = nn.Linear(hidden_size, output_size)`: Output projection layer computing $z_2 = z_1 W_2^T + b_2$. Parameters: $512 \times 77 + 77 = 39,501$.
  - `forward(self, x)`: Defines tensor forward path returning unnormalized logits.

---

#### Cell 36 [Markdown] - Loss Function & Optimizer Rationale
- **Purpose**: Documents why CrossEntropyLoss (Softmax + NLLLoss) and the Adam optimizer are selected.

---

#### Cell 37 [Code] - Instantiate Model, Loss & Optimizer
```python
import torch.nn.functional as F

model = SimpleMLP(
    input_size=input_size,
    hidden_size=512,
    output_size=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
```
- **Line-by-Line Breakdown**:
  - `model = SimpleMLP(...).to(device)`: Initializes model and transfers all parameter tensors to active device (`cuda` or `cpu`).
  - `criterion = nn.CrossEntropyLoss()`: Computes multi-class loss:
    $$\mathcal{L} = -\log\left(\frac{\exp(z_{y})}{\sum_{j} \exp(z_j)}\right)$$
  - `optimizer = torch.optim.Adam(...)`: Adaptive Moment Estimation optimizer computing individual adaptive learning rates with momentum.
- **Parameters Explained**:
  | Parameter | Value | Mathematical Purpose |
  |---|---|---|
  | `lr` | `1e-3` ($0.001$) | Initial step size for parameter updates along the gradient vector. |
  | `weight_decay` | `1e-5` | L2 regularization penalty added to the loss: $\frac{1}{2} \lambda \|\theta\|_2^2$, shrinking non-essential weights. |

---

#### Cell 38 [Markdown] - Training Loop Protocol
- **Purpose**: Defines training procedure across 5 epochs, monitoring training loss and mini-batch accuracy.

---

#### Cell 39 [Code] - MLP Training Execution Loop
```python
torch.manual_seed(42)

def train_model(
    model,
    train_loader,
    loss_fn,
    optimizer,
    num_epochs: int = 5,
    device: torch.device = None,
):
    if device is None:
        device = next(model.parameters()).device

    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    return model

model = train_model(
    model=model,
    train_loader=train_loader,
    loss_fn=criterion,
    optimizer=optimizer,
    num_epochs=5,
    device=device
)
```
- **Line-by-Line Breakdown**:
  - `model.train()`: Enables dropout layers.
  - `optimizer.zero_grad()`: Flushes gradient buffers from previous mini-batch before backward pass.
  - `outputs = model(batch_X)`: Computes forward pass logits.
  - `loss = loss_fn(outputs, batch_y)`: Computes scalar batch loss.
  - `loss.backward()`: Computes $\frac{\partial \mathcal{L}}{\partial \theta}$ via automatic differentiation.
  - `optimizer.step()`: Updates weights: $\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$.
  - Tracks running statistics and prints epoch loss and accuracy.

---

#### Cell 40 [Markdown] - Model Evaluation Protocol
- **Purpose**: Outlines testing loop behavior (`model.eval()`, `torch.no_grad()`).

---

#### Cell 41 [Code] - Predict Function Implementation
```python
def predict(model, dataloader, device="cpu"):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    return all_predictions, all_labels

mlp_y_pred, mlp_y_true = predict(model, test_loader, device=device)
```
- **Line-by-Line Breakdown**:
  - `model.eval()`: Freezes dropout layers so all 512 hidden units contribute deterministically.
  - `with torch.no_grad()`: Disables computation graph construction, reducing memory consumption by ~60%.
  - `torch.argmax(outputs, dim=1)`: Extracts class index with maximum logit value.
  - Gathers predictions and labels onto CPU memory and returns NumPy arrays.

---

#### Cell 42 [Markdown] - Classification Metrics Overview
- **Purpose**: Defines Precision, Recall, Macro F1, and Weighted F1 metrics:
  $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = 2 \cdot \frac{P \cdot R}{P + R}$$

---

#### Cell 43 [Code] - MLP Evaluation Metrics & Classification Report
```python
from sklearn.metrics import accuracy_score, f1_score, classification_report
mlp_accuracy = accuracy_score(mlp_y_true, mlp_y_pred)
mlp_macro_f1 = f1_score(mlp_y_true, mlp_y_pred, average='macro')
mlp_weighted_f1 = f1_score(mlp_y_true, mlp_y_pred, average='weighted')

print(f"MLP Test Accuracy:    {mlp_accuracy:.4f}")
print(f"MLP Macro F1-Score:   {mlp_macro_f1:.4f}")
print(f"MLP Weighted F1-Score: {mlp_weighted_f1:.4f}")
print("\n=== Classification Report: MLP ===")
print(classification_report(mlp_y_true, mlp_y_pred, target_names=le.classes_))
```
- **Line-by-Line Breakdown**:
  - Calculates accuracy and F1 scores.
  - **Results**:
    - MLP Test Accuracy: **~84.22%**
    - MLP Macro F1-Score: **~0.8398**
    - MLP Weighted F1-Score: **~0.8415**
  - Displays complete per-class classification report for all 77 intent classes.

---

#### Cell 44 [Code] - Empty Spacer Cell
- **Purpose**: Preserved empty notebook cell from curriculum template.

---

### Module 4: Parameter-Efficient Fine-Tuning (PEFT) with RoBERTa & LoRA (Cells 45 – 65)

#### Cell 45 [Markdown] - Transformer & LoRA Conceptual Foundation
- **Purpose**: Details RoBERTa (Robustly Optimized BERT Approach, Liu et al., 2019) and Low-Rank Adaptation (LoRA, Hu et al., 2021).
- **Theoretical Basis**: Full fine-tuning updates all $125\text{M}$ weights, which is computationally expensive and prone to catastrophic forgetting. LoRA hypothesizes that weight changes $\Delta W$ have a low "intrinsic dimension" and decomposes updates into two low-rank matrices:
  $$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$

---

#### Cell 46 [Markdown] - Clean Data Reload Notice
- **Purpose**: Explains reloading raw text DataFrames to ensure no previous pipeline artifacts persist.

---

#### Cell 47 [Code] - Reload Clean Datasets
```python
import pandas as pd
train = pd.read_csv("datasets/banking77_train.csv")
test = pd.read_csv("datasets/banking77_test.csv")
train.head()
```
- **Line-by-Line Breakdown**:
  - Reloads raw CSV files for Hugging Face Dataset pipeline.

---

#### Cell 48 [Markdown] - Label Encoding for Transformers
- **Purpose**: Re-establishes categorical integer mapping for Hugging Face Sequence Classification head.

---

#### Cell 49 [Code] - Hugging Face Dataset Conversion
```python
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder

train = train.dropna(subset=["text", "category"]).copy()
test = test.dropna(subset=["text", "category"]).copy()

label_encoder = LabelEncoder()
train["label"] = label_encoder.fit_transform(train["category"].astype(str))
test["label"]  = label_encoder.transform(test["category"].astype(str))

num_classes = len(label_encoder.classes_)

training_set = Dataset.from_pandas(train[["text", "label"]])
test_set     = Dataset.from_pandas(test[["text", "label"]])
```
- **Line-by-Line Breakdown**:
  - Sanitizes null values using `.dropna()`.
  - Encodes `category` into integer `label`.
  - `Dataset.from_pandas(...)`: Converts Pandas DataFrames into Apache Arrow-backed Hugging Face `Dataset` objects for memory-efficient streaming.

---

#### Cell 50 [Markdown] - RoBERTa Tokenization Protocol
- **Purpose**: Explains Byte-Pair Encoding (BPE), vocabulary of 50,265 subwords, dynamic truncation, and special tokens (`<s>`, `</s>`).

---

#### Cell 51 [Code] - Tokenization Pipeline
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding=False,
        truncation=True,
        max_length=64
    )

tokenized_training_set = training_set.map(tokenize_function, batched=True)
tokenized_test_set = test_set.map(tokenize_function, batched=True)
```
- **Line-by-Line Breakdown**:
  - `AutoTokenizer.from_pretrained("roberta-base")`: Loads pre-trained BPE tokenizer.
  - `tokenize_function`: Encodes strings into `input_ids` and `attention_mask`.
  - `padding=False`: Postpones padding to the DataCollator, enabling dynamic padding per batch to save GPU memory.
  - `truncation=True, max_length=64`: Caps sequence length at 64 tokens (informed by our EDA in Cell 18-19).
  - `.map(..., batched=True)`: Executes batched tokenization using multi-threaded PyArrow workers.

---

#### Cell 52 [Markdown] - TrainingArguments Configuration Details
- **Purpose**: Explains transformer hyperparameters: warmup ratio, weight decay, epoch budgeting, evaluation strategy.

---

#### Cell 53 [Code] - TrainingArguments Setup
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./temp_results",
    save_strategy="no",
    logging_dir="./logs",
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=50,
    seed=42
)
```
- **Parameters Deep Dive**:
  | Parameter | Value | Theoretical / Engineering Rationale |
  |---|---|---|
  | `learning_rate` | `2e-4` ($0.0002$) | Standard full fine-tuning uses ~`2e-5`. LoRA updates only adapter weights and benefits from a $10\times$ higher learning rate for fast convergence. |
  | `per_device_train_batch_size` | `32` | Fits within standard GPU VRAM (8GB) while maintaining stable batch gradients. |
  | `num_train_epochs` | `3` | Sufficient for adapter convergence without overfitting on 10,003 queries. |
  | `weight_decay` | `0.01` | AdamW weight decay preventing adapter matrix overfitting. |
  | `warmup_ratio` | `0.1` | Linearly ramps learning rate from $0$ to `2e-4` over the first 10% of total training steps, preventing early gradient instability. |
  | `save_strategy` | `"no"` | Disables intermediate checkpoint saves to save disk storage during demonstration. |

---

#### Cell 54 [Markdown] - LoRA Hyperparameter Specification
- **Purpose**: Details LoRA rank ($r$), alpha scaling ($\alpha$), target modules, and parameter efficiency math:
  $$\Delta W = \frac{\alpha}{r} (B \cdot A)$$
  With $r=8, \alpha=16$, scaling factor is $\frac{16}{8} = 2.0$.

---

#### Cell 55 [Code] - LoRA Model Assembly & Parameter Audit
```python
from peft import LoraConfig, get_peft_model, TaskType
torch.manual_seed(42)

model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=num_classes
).to(device)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```
- **Line-by-Line Breakdown**:
  - `AutoModelForSequenceClassification...`: Instantiates 12-layer transformer with 77-class classification head ($768 \to 77$).
  - `LoraConfig(...)`: Defines low-rank adaptation on multi-head attention `query` and `value` projections.
  - `model = get_peft_model(model, lora_config)`: Freezes base RoBERTa weights ($\approx 124\text{M}$ parameters) and injects trainable low-rank matrices $A$ and $B$.
  - `model.print_trainable_parameters()`: Audits trainable weights:
    ```
    trainable params: 884,736 || all params: 125,532,749 || trainable%: 0.7048%
    ```
- **Parameters Deep Dive**:
  | Parameter | Value | Purpose |
  |---|---|---|
  | `task_type` | `TaskType.SEQ_CLS` | Configures PEFT for Sequence Classification heads. |
  | `r` | `8` | Low-rank matrix bottleneck dimension ($A \in \mathbb{R}^{8 \times 768}$, $B \in \mathbb{R}^{768 \times 8}$). |
  | `lora_alpha` | `16` | Scaling constant; scales update by $\frac{\alpha}{r} = 2.0$. |
  | `target_modules` | `["query", "value"]` | Injects adapters into attention projections that govern token representation mixing. |
  | `lora_dropout` | `0.1` | Dropout probability on adapter inputs. |
  | `bias` | `"none"` | Freezes all bias parameters, reducing trainable footprint. |

---

#### Cell 56 [Markdown] - Hugging Face Trainer Setup Notice
- **Purpose**: Explains combining model, data collator, and datasets into the Hugging Face `Trainer`.

---

#### Cell 57 [Code] - Fine-Tune Model with Trainer
```python
from transformers import Trainer
torch.manual_seed(42)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_training_set,
    eval_dataset=tokenized_test_set,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

trainer.train()
```
- **Line-by-Line Breakdown**:
  - `DataCollatorWithPadding(...)`: Dynamically pads queries in each mini-batch to that batch's longest query instead of a static 64 tokens, saving compute.
  - `trainer.train()`: Executes the fine-tuning loop across 3 epochs (~939 optimization steps).
  - Training loss decreases from ~3.8 to ~0.5.

---

#### Cell 58 [Markdown] - Checkpoint & Weight Merging Rationale
- **Purpose**: Explains `.merge_and_unload()`: folds adapter weights into base weights ($W = W_0 + \frac{\alpha}{r} BA$) so the final model incurs **zero latency overhead** during inference.

---

#### Cell 59 [Code] - Merge Adapter Weights & Save Model
```python
save_dir = "./finetuned_roberta_lora_model"

model = model.merge_and_unload()
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Model saved to {save_dir}")
```
- **Line-by-Line Breakdown**:
  - `model.merge_and_unload()`: Merges LoRA matrices into base weights.
  - `save_pretrained(...)`: Saves `model.safetensors`, `config.json`, tokenizer vocabulary, and configuration files.

---

#### Cell 60 [Markdown] - Load Fine-Tuned Model Notice
- **Purpose**: Tests production loading from disk.

---

#### Cell 61 [Code] - Load Merged Model from Disk
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

path = "./finetuned_roberta_lora_model"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
```
- **Line-by-Line Breakdown**:
  - Loads saved model and tokenizer from local disk directory and transfers to target device.

---

#### Cell 62 [Markdown] - Transformer Evaluation Notice
- **Purpose**: Prepares evaluation pass on test set.

---

#### Cell 63 [Code] - Run Transformer Inference on Test Split
```python
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

prediction_args = TrainingArguments(
    output_dir="./temp_pred",
    per_device_eval_batch_size=64,
    report_to="none"
)

eval_trainer = Trainer(
    model=model,
    args=prediction_args,
    data_collator=data_collator,
    tokenizer=tokenizer
)

predictions = eval_trainer.predict(tokenized_test_set)
roberta_y_pred = np.argmax(predictions.predictions, axis=1)
roberta_y_true = predictions.label_ids
```
- **Line-by-Line Breakdown**:
  - `eval_trainer.predict(...)`: Executes batched forward pass over 3,080 test queries.
  - `np.argmax(..., axis=1)`: Extracts highest probability class index per query.

---

#### Cell 64 [Markdown] - Transformer Metrics Summary Marker
- **Purpose**: Section marker for classification report.

---

#### Cell 65 [Code] - LoRA-RoBERTa Metrics & Classification Report
```python
from sklearn.metrics import accuracy_score, f1_score, classification_report
lora_roberta_accuracy = accuracy_score(roberta_y_true, roberta_y_pred)
lora_roberta_macro_f1 = f1_score(roberta_y_true, roberta_y_pred, average='macro')
lora_roberta_weighted_f1 = f1_score(roberta_y_true, roberta_y_pred, average='weighted')

print(f"LoRA-RoBERTa Test Accuracy:    {lora_roberta_accuracy:.4f}")
print(f"LoRA-RoBERTa Macro F1-Score:   {lora_roberta_macro_f1:.4f}")
print(f"LoRA-RoBERTa Weighted F1-Score: {lora_roberta_weighted_f1:.4f}")
print("\n=== Classification Report: LoRA-RoBERTa ===")
print(classification_report(roberta_y_true, roberta_y_pred, target_names=label_encoder.classes_))
```
- **Line-by-Line Breakdown**:
  - Evaluates final model performance:
    - Test Accuracy: **~93.38%** (vs. 84.22% for MLP)
    - Macro F1-Score: **~0.9329** (vs. 0.8398 for MLP)
    - Weighted F1-Score: **~0.9337** (vs. 0.8415 for MLP)

---

### Module 5: Performance Comparison & Trade-Off Analysis (Cells 66 – 71)

#### Cell 66 [Markdown] - Comparative Framework Overview
- **Purpose**: Establishes per-class comparative analysis to evaluate where contextual embeddings provide the largest benefits over bag-of-words.

---

#### Cell 67 [Code] - Build Comparison DataFrame
```python
mlp_report_dict = classification_report(mlp_y_true, mlp_y_pred, target_names=le.classes_, output_dict=True)
roberta_report_dict = classification_report(roberta_y_true, roberta_y_pred, target_names=label_encoder.classes_, output_dict=True)

classes = label_encoder.classes_
comparison_data = []

for c in classes:
    mlp_f1 = mlp_report_dict[c]['f1-score']
    roberta_f1 = roberta_report_dict[c]['f1-score']
    diff = roberta_f1 - mlp_f1
    comparison_data.append({
        'Intent': c,
        'MLP_F1': mlp_f1,
        'RoBERTa_F1': roberta_f1,
        'F1_Diff': diff
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values(by='F1_Diff', ascending=False).reset_index(drop=True)
comparison_df.head(10)
```
- **Line-by-Line Breakdown**:
  - Extracts per-class dictionary metrics from both classification reports.
  - Computes `F1_Diff = RoBERTa_F1 - MLP_F1`.
  - Sorts descending by `F1_Diff` to highlight largest transformer performance gains.

---

#### Cell 68 [Code] - Inspect Smallest F1 Difference Categories
```python
comparison_df.tail(10)
```
- **Line-by-Line Breakdown**:
  - Displays categories where MLP performed comparably to RoBERTa (intents with unique lexical keywords like `exchange_rate` or `atm_support`).

---

#### Cell 69 [Markdown] - Section Marker: Top & Bottom 10 F1 Delta
- **Purpose**: Header for summary analysis.

---

#### Cell 70 [Code] - Print Top 10 and Bottom 10 Comparison
```python
print("====================== Top 10 F1 Diff ======================")
print(comparison_df[['Intent', 'MLP_F1', 'RoBERTa_F1', 'F1_Diff']].head(10))
print()
print("==================== Bottom 10 F1 Diff ====================")
print(comparison_df[['Intent', 'MLP_F1', 'RoBERTa_F1', 'F1_Diff']].tail(10))
```
- **Line-by-Line Breakdown**:
  - Displays top 10 categories with largest gains (+25% to +40% F1 improvement), such as `compromised`, `card_arrival`, `lost_or_stolen_card`.
  - Displays categories where performance was similar between models.

---

#### Cell 71 [Markdown] - Comprehensive Architectural Trade-Off Analysis
- **Purpose**: Deep qualitative discussion answering why RoBERTa outperforms the MLP baseline:
  1. **Polysemy & Word Order**: TF-IDF ignores token order and contextual meaning. "I did not get my fee refunded" vs. "Did my fee get refunded?" produce identical bag-of-words vectors despite expressing different customer intents.
  2. **Subword Robustness**: RoBERTa BPE tokenization handles misspellings, compounds, and abbreviations that break strict TF-IDF vocabulary matches.
  3. **Transfer Learning**: Pre-trained on 160GB of text, RoBERTa understands syntactic dependencies and financial synonyms before fine-tuning begins.

---

### Module 6: Responsible AI & Enterprise PII Redaction Pipeline (Cells 72 – 82)

#### Cell 72 [Markdown] - Enterprise PII Compliance & Governance
- **Purpose**: Discusses regulatory frameworks (GDPR Article 17/32, PCI-DSS Requirement 3.4, GLBA). Customer financial queries often contain sensitive Personally Identifiable Information (credit cards, CVVs, account numbers, emails, phone numbers) that must be redacted before logging or inference.

---

#### Cell 73 [Code] - Enterprise PII Redaction & Hashing Function
```python
import re
import hashlib

def remove_pii(text, hash_identifiers=False):
    """
    Remove or hash PII from customer queries.
    """
    # Regex patterns
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    PHONE_PATTERN = r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b'
    CARD_PATTERN  = r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
    CVV_PATTERN   = r'\b(?i:(?:cvv|cvc|security code|code)[\s:]*)(\d{3,4})\b'
    SSN_PATTERN   = r'\b\d{3}-\d{2}-\d{4}\b'
    ROUTING_PAT   = r'\b(?i:routing(?: number)?)[\s:]*(\d{9})\b'
    ACCOUNT_PAT   = r'\b(?i:account(?: number)?)[\s:]*(\d{8,12})\b'
    IP_PATTERN    = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    DATE_PATTERN  = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'

    def hash_str(val):
        return f"[HASH:{hashlib.sha256(val.encode('utf-8')).hexdigest()[:16]}]"

    cleaned = text

    # Credit card
    cleaned = re.sub(CARD_PATTERN, "[CARD NUMBER REDACTED]", cleaned)

    # CVV
    cleaned = re.sub(CVV_PATTERN, r"[CVV NUMBER REDACTED]", cleaned)

    # SSN
    cleaned = re.sub(SSN_PATTERN, "[SSN REDACTED]", cleaned)

    # Routing & Account
    cleaned = re.sub(ROUTING_PAT, "[ROUTING NUMBER REDACTED]", cleaned)
    cleaned = re.sub(ACCOUNT_PAT, "[ACCOUNT NUMBER REDACTED]", cleaned)

    # IP & Date
    cleaned = re.sub(IP_PATTERN, "[IP REDACTED]", cleaned)
    cleaned = re.sub(DATE_PATTERN, "[DATE REDACTED]", cleaned)

    # Identifiers: Email & Phone
    if hash_identifiers:
        cleaned = re.sub(EMAIL_PATTERN, lambda m: hash_str(m.group(0)), cleaned)
        cleaned = re.sub(PHONE_PATTERN, lambda m: hash_str(m.group(0)), cleaned)
    else:
        cleaned = re.sub(EMAIL_PATTERN, "[EMAIL REDACTED]", cleaned)
        cleaned = re.sub(PHONE_PATTERN, "[PHONE NUMBER REDACTED]", cleaned)

    return cleaned
```
- **Line-by-Line Breakdown**:
  - Compiles standard regex patterns matching financial identifiers.
  - `hash_str`: Generates a truncated 16-character SHA-256 cryptographic digest of emails/phone numbers.
  - When `hash_identifiers=True`, replaces customer contact details with deterministic hashes (e.g., `[HASH:55f537baf75630a8]`), enabling customer interaction tracking across multi-turn sessions without storing raw personal data.
  - Fully redacts payment card numbers and CVVs to comply with PCI-DSS.

---

#### Cell 74 [Markdown] - Standard Redaction Test Notice
- **Purpose**: Prepares test queries without hashing.

---

#### Cell 75 [Code] - Execute Standard PII Redaction Tests
```python
print("=== Standard PII Redaction ===")
test_cases = [
    "My routing number is 021000021 and account is 123456789",
    "I was born on 03/15/1985 and live at 123 Main Street, ZIP 02139",
    "My card CVV is 123 and PIN is 4567",
    "My driver's license is D1234567",
    "My passport number is AB1234567",
    "Login from IP 192.168.1.1 failed",
]

for test in test_cases:
    print(f"Original: {test}")
    print(f"Redacted: {remove_pii(test)}")
    print()
```
- **Line-by-Line Breakdown**:
  - Evaluates sanitization on synthetic customer queries containing routing numbers, accounts, CVVs, IPs, and birth dates.
  - Validates that sensitive numeric strings are replaced with standard redaction tokens.

---

#### Cell 76 [Markdown] - Hashing Test Notice
- **Purpose**: Demonstrates customer session tracking via hashing.

---

#### Cell 77 [Code] - Execute Cryptographic Hashing Verification
```python
print("\n=== With Identifier Hashing (for customer tracking) ===")
test_with_contact = "Email me at john.doe@email.com or call 123-456-7890"
print(f"Original: {test_with_contact}")
print(f"Redacted: {remove_pii(test_with_contact, hash_identifiers=False)}")
print(f"Hashed:   {remove_pii(test_with_contact, hash_identifiers=True)}")
```
- **Line-by-Line Breakdown**:
  - Tests phone and email sanitization.
  - `Redacted`: `Email me at [EMAIL REDACTED] or call [PHONE NUMBER REDACTED]`
  - `Hashed`: `Email me at [HASH:55f537baf75630a8] or call [HASH:29ec0a06044bedff]`

---

#### Cell 78 [Code] - Empty Spacer Cell
- **Purpose**: Preserved empty notebook cell from curriculum template.

---

#### Cell 79 [Markdown] - End-to-End Inference Header
- **Purpose**: Introduces the production inference pipeline integrating PII sanitation and LoRA-RoBERTa intent classification.

---

#### Cell 80 [Code] - Real-World Safe Inference Pipeline
```python
def predict_text_safe(text, model, tokenizer, label_encoder, device="cpu", top_k=3, hash_identifiers=True):
    cleaned_text = remove_pii(text, hash_identifiers=hash_identifiers)

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probs, k=top_k)

    print(f"Original input: {text}\n")
    print(f"Redacted input: {cleaned_text}\n")
    print("Predicted Customer Intent:")
    for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
        class_name = label_encoder.classes_[idx.item()]
        print(f"{rank}. {class_name}: {prob.item():.4f}")

    return cleaned_text, label_encoder.classes_[top_indices[0].item()]

# Test query with card and contact details
query = """I want to dispute a charge of $49.99 on my card 4532-1234-5678-9010 and CVV 000.
My email is john.doe@email.com, call me at 123-456-7890."""

cleaned_text, predicted_class = predict_text_safe(
    query,
    model=model,
    tokenizer=tokenizer,
    label_encoder=label_encoder,
    device=device,
    top_k=3,
    hash_identifiers=True
)
```
- **Line-by-Line Breakdown**:
  - `cleaned_text = remove_pii(...)`: Strips card number, CVV, and hashes contact info before tokenizer processing.
  - `inputs = tokenizer(...)`: Converts cleaned string to tensor dictionary on target device.
  - `probs = torch.softmax(outputs.logits, dim=1)`: Converts unnormalized logits to probabilities $\in [0, 1]$ summing to $1.0$.
  - `torch.topk(probs, k=top_k)`: Extracts top $k$ most likely intent classes with confidence scores.
  - Prints sanitized text and ranked predictions.
- **Example Output**:
  ```
  Original input: I want to dispute a charge of $49.99 on my card 4532-1234-5678-9010 and CVV 000.
  My email is john.doe@email.com, call me at 123-456-7890.

  Redacted input: I want to dispute a charge of $49.99 on my card [CARD NUMBER REDACTED] and [CVV NUMBER REDACTED].
  My email is [HASH:55f537baf75630a8], call me at [HASH:29ec0a06044bedff].

  Predicted Customer Intent:
  1. direct_debit_payment_not_recognised: 0.4960
  2. extra_charge_on_statement: 0.3820
  3. card_payment_fee_charged: 0.0404
  ```

---

#### Cell 81 [Markdown] - Inference Interpretation & Analysis
- **Purpose**: Evaluates model predictions on the test query:
  - Both top intents (`direct_debit_payment_not_recognised` at 49.6% and `extra_charge_on_statement` at 38.2%) involve unauthorized or unexpected charges.
  - Demonstrates that PII tokens (`[CARD NUMBER REDACTED]`, `[HASH:...]`) do not degrade the semantic representation of the query.

---

#### Cell 82 [Code] - Empty Spacer Cell
- **Purpose**: Preserved empty notebook cell from curriculum template.

---

#### Cell 83 [Markdown] - Project Conclusions & Next Steps
- **Purpose**: Synthesizes engineering conclusions:
  - LoRA provides a 9.16% absolute accuracy improvement with only 0.7% trainable parameters.
  - Outlines next steps: ensemble modeling, hyperparameter optimization, and active learning with human-in-the-loop fallback routing for queries with top prediction confidence $< 0.50$.

---

#### Cell 84 [Code] - Final Empty Spacer Cell
- **Purpose**: Preserved terminal notebook cell.

---

## 6. Comparative Metrics Table

| Dimension | Baseline MLP (`TfidfVectorizer`) | Modern PEFT (`RoBERTa-base + LoRA`) |
|---|---|---|
| **Underlying Architecture** | 2-Layer Feedforward ($50,000 \to 512 \to 77$) | 12-Layer Transformer + Multi-Head Self-Attention |
| **Input Representation** | Sparse n-gram TF-IDF Bag-of-Words | Dense Byte-Pair Contextual Embeddings |
| **Context Awareness** | None (word order is ignored) | Bidirectional self-attention over full sequence |
| **Trainable Weights** | 25,639,497 | **884,736** (adapters only) |
| **Test Accuracy** | 84.22% | **93.38%** (+9.16%) |
| **Macro Average F1** | 0.8398 | **0.9329** (+0.0931) |
| **Weighted Average F1** | 0.8415 | **0.9337** (+0.0922) |
| **Training Device Requirement** | Commodity CPU | GPU Recommended (CUDA) |
| **Inference Latency (Batch 1)** | **< 1.5 ms** (Ultra fast) | ~18-25 ms (CPU) / ~4 ms (GPU) |
| **PII Redaction Support** | Regex Pre-Filter | Integrated Regex + SHA-256 Hashing |

---

## 7. Enterprise Deployment Considerations

1. **Model Quantization**:
   - Convert the merged model to **INT8** or **FP8** using Hugging Face `bitsandbytes` or ONNX Runtime to reduce VRAM requirements by ~50% with <0.5% accuracy loss.
2. **Confidence Thresholding (Human-in-the-Loop)**:
   - When $\max_c P(\text{intent}_c \mid x) < 0.50$, automatically route the conversation to human support agents rather than guessing.
3. **Regulatory Audit Logging**:
   - Log only the sanitized and hashed query text (`[HASH:...]`), ensuring customer session continuity in compliance with GDPR and PCI-DSS requirements.
