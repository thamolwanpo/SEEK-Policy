# SEEK-Policy: Cross-Modal Retrieval for Context-Aware Climate Policy Recommendation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

SEEK-Policy (Semantic Embeddings for Environment & Knowledge-based Policy) is an intelligent expert system designed to assist climate policy decision-making by integrating multivariate time-series data with policy document retrieval. The framework uses Retrieval-Augmented Generation (RAG) to align real-world environmental trends with semantically relevant climate policy documents.

### Key Features

-   **TimeTranscriber**: Transforms multivariate time-series signals into policy-relevant summaries using LLMs
-   **ChunkAlign**: Performs fine-grained retrieval by mapping summaries to semantically aligned policy document segments
-   **PolicySynthesizer**: Composes retrieved content into coherent, interpretable recommendations
-   **POLiMATCH Dataset**: Links 4,000+ climate policy documents with 10 years of time-series data across multiple countries and sectors

### Performance Highlights

SEEK-Policy achieves significant improvements over baseline methods:

-   **Hit@1**: 0.76 (vs. 0.50 human summary baseline)
-   **Hit@5**: 0.87
-   **MRR@5**: 0.81

## Table of Contents

-   [Installation](#installation)
-   [Dataset](#dataset)
-   [Project Structure](#project-structure)
-   [Usage](#usage)
-   [Methodology](#methodology)
-   [Baselines](#baselines)
-   [Evaluation](#evaluation)
-   [Requirements](#requirements)
-   [Citation](#citation)
-   [License](#license)
-   [Acknowledgments](#acknowledgments)

## Installation

### Prerequisites

-   Python 3.8 or higher
-   Neo4j Database (for knowledge graph storage)
-   OpenAI API key (for LLM-based components)

### Setup Steps

1. **Clone the repository**

```bash
git clone https://github.com/thamolwanpo/SEEK-Policy.git
cd SEEK-Policy
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up Neo4j Database**

-   Download and install [Neo4j](https://neo4j.com/download/)
-   Create a new database instance
-   Note your connection credentials

4. **Configure environment variables**

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

## Dataset

### POLiMATCH Dataset

The POLiMATCH (Policy and Linked Multimodal Alignment of Time-Series and Climate Histories) dataset integrates:

-   **Policy Documents**: 4,534 climate policy documents from Climate Policy Radar
    -   1,243 with English summaries
    -   Metadata: sectors, instruments, countries, timelines
-   **Time-Series Data**: 127 curated tables from Our World In Data
    -   34 climate-related datasets
    -   50 socio-economic datasets
    -   43 other relevant topics (energy, agriculture, waste)
    -   Annual data for 200+ countries

### Download Dataset

The preprocessed dataset can be downloaded from:

-   [Google Drive Link](https://drive.google.com/drive/folders/1zqR8mDpkETPYYBnQxDl7YGweSMzL7Z6U?usp=sharing)

### Data Sources

-   **Climate Policy Radar**: [climatepolicyradar.org](https://www.climatepolicyradar.org/)
-   **Our World In Data**: [ourworldindata.org](https://ourworldindata.org/)

For updated versions, please visit the respective websites or use the [owid-catalog](https://pypi.org/project/owid-catalog/) Python library.

## Project Structure

```
SEEK-Policy/
├── data/                          # Dataset files and README
│   └── README.md                  # Data download instructions
├── scripts/                       # Python scripts for reproducibility
│   ├── 0_split_test_set.py       # Test set creation with stratified sampling
│   ├── 0_visualize_data_distribution.py  # Data visualization
│   ├── 1_time_series_data_preprocessing.py  # Time-series preprocessing
│   ├── 2_data_preparation_and_time_series_scaling.py  # Data preparation
│   ├── 3_Generate_RAG_Summary.py  # SEEK-Policy summary generation
│   ├── 3_generate_role_agents_summary.py  # Role-based baseline
│   ├── 3_train_mlp.py            # Siamese network baseline training
│   └── 4_kg_retriever.py         # Knowledge graph construction and retrieval
├── notebooks/                     # Jupyter notebook versions of scripts
├── README.md                      # This file
├── requirements.txt               # Python dependencies
└── LICENSE                        # MIT License
```

## Usage

### Step 1: Prepare the Dataset

#### Split Test Set

```bash
python scripts/0_split_test_set.py
```

Creates a stratified test set of 100 documents ensuring:

-   Geographic diversity across regions
-   No duplicate countries
-   Balanced sector representation

#### Visualize Data Distribution

```bash
python scripts/0_visualize_data_distribution.py
```

Generates distribution plots for regions, countries, and sectors.

### Step 2: Preprocess Time-Series Data

```bash
python scripts/1_time_series_data_preprocessing.py
```

This script:

-   Loads metadata from Our World In Data
-   Filters columns (removes non-numeric, constant, highly correlated)
-   Handles missing values via interpolation and backfilling
-   Merges country-level time-series data
-   Saves processed data for each country

### Step 3: Data Preparation and Scaling

```bash
python scripts/2_data_preparation_and_time_series_scaling.py
```

Operations:

-   Cleans policy summaries (removes HTML tags)
-   Filters by geography and sector
-   Creates 10-year historical windows for each policy
-   One-hot encodes sectors
-   Standardizes features using StandardScaler
-   Generates training data in JSONL format

### Step 4: Generate Summaries

#### SEEK-Policy (RAG-based)

```bash
python scripts/3_Generate_RAG_Summary.py
```

Pipeline:

1. **TimeTranscriber**: Selects relevant time-series tables and generates summaries
2. **ChunkAlign**: Retrieves semantically similar policy chunks from Neo4j
3. **PolicySynthesizer**: Composes final policy recommendation

#### Role-Based Multi-Agent Baseline

```bash
python scripts/3_generate_role_agents_summary.py
```

Simulates expert advisors:

-   Climate policy advisor
-   Socio-economic advisor
-   Sustainable development advisor
-   Policy summarizer

#### Siamese Network Baseline

```bash
python scripts/3_train_mlp.py
```

Trains a contrastive learning model with triplet loss to embed time-series and policy text in a shared space.

### Step 5: Knowledge Graph Construction and Retrieval

```bash
python scripts/4_kg_retriever.py
```

Features:

-   Constructs Neo4j knowledge graph from policy documents
-   Creates vector embeddings for semantic search
-   Implements both structured (Cypher) and unstructured (vector) retrieval
-   Evaluates retrieval performance with Hit@K metrics

## Methodology

### Architecture

The SEEK-Policy framework consists of three core modules:

1. **TimeTranscriber**

    - Input: Multivariate time-series data + sector metadata
    - Process: LLM-based transformation to natural language
    - Output: Policy-relevant summary

2. **ChunkAlign**

    - Input: Generated summary from TimeTranscriber
    - Process: Semantic embedding + cosine similarity search
    - Output: Top-k relevant policy chunks

3. **PolicySynthesizer**
    - Input: Retrieved policy chunks
    - Process: LLM-based composition
    - Output: Unified policy recommendation

### Key Algorithms

**Semantic Retrieval:**

```
Sim(v_q, v_c) = (v_q · v_c) / (||v_q|| ||v_c||)
```

where v_q is the query embedding and v_c is the chunk embedding.

**Document Ranking:**
Documents are scored based on aggregated similarity of their retrieved chunks, with deduplication to ensure diversity.

## Baselines

### 1. Climate Policy Radar (Keyword Search)

-   Static keyword-based retrieval
-   Hit@5: 0.50, MRR@5: 0.38

### 2. Siamese Network with Contrastive Learning

-   Dual-encoder architecture
-   Triplet loss training
-   Hit@5: 0.04, MRR@5: 0.01

### 3. Role-Based Multi-Agent System

-   Zero-shot LLM prompting
-   Domain-specific personas
-   Hit@5: 0.34, MRR@5: 0.27

### 4. Human Summary Baseline

-   Expert-written policy summaries
-   Hit@5: 0.70, MRR@5: 0.58

## Evaluation

### Metrics

-   **Hit@k**: Proportion of queries where correct document appears in top-k results
-   **Mean Reciprocal Rank (MRR)**: Average inverse rank of first correct result

### Filtering Scenarios

The framework supports four filtering modes:

-   **No filter**: Full corpus search
-   **Sector filter**: Matches policy sector
-   **Country filter**: Matches geographic region
-   **Sector + Country filter**: Combined filtering

### Results Summary

| Model                | Hit@1    | Hit@5    | MRR@5    |
| -------------------- | -------- | -------- | -------- |
| Climate Policy Radar | 0.31     | 0.50     | 0.38     |
| Human Summary        | 0.50     | 0.70     | 0.58     |
| Siamese Network      | 0.00     | 0.04     | 0.01     |
| Role-Based Agents    | 0.21     | 0.34     | 0.27     |
| **SEEK-Policy**      | **0.76** | **0.87** | **0.81** |

## Requirements

### Core Dependencies

```
python>=3.8
torch>=2.0.0
pytorch-lightning>=2.0.0
transformers>=4.30.0
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-community>=0.0.20
neo4j>=5.0.0
openai>=1.0.0
sentence-transformers>=2.2.0
```

### Data Processing

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
dask[dataframe]>=2023.5.0
```

### Visualization

```
matplotlib>=3.7.0
seaborn>=0.12.0
geopandas>=0.13.0
```

### Additional Tools

```
beautifulsoup4>=4.12.0
python-dotenv>=1.0.0
tqdm>=4.65.0
```

See `requirements.txt` for complete list with version specifications.

## Citation

If you use SEEK-Policy or the POLiMATCH dataset in your research, please cite:

```bibtex
@misc{seekpolicy2025,
  title={SEEK-Policy: Cross-Modal Retrieval for Context-Aware Climate Policy Recommendation Using Semantic Embeddings},
  author={Poopradubsil, Thamolwan and Chen, Sheng-Chih and Latcharote, Panon and Thaipisutikul, Tipajin},
  year={2025},
  note={Submitted for publication}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, issues, or collaboration opportunities:

-   GitHub Issues: [https://github.com/thamolwanpo/SEEK-Policy/issues](https://github.com/thamolwanpo/SEEK-Policy/issues)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Note**: Both `scripts/` and `notebooks/` folders contain the same implementation - `scripts/` contains Python scripts for production use, while `notebooks/` contains Jupyter notebook versions for interactive exploration and development.
