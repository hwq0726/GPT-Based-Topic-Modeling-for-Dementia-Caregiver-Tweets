# GPT-Based Topic Modeling for Dementia Caregiver Tweets

This repository contains code for evaluating GPT-based topic modeling approaches against traditional baseline methods for analyzing tweets from dementia caregivers. The research compares the semantic coherence and topic quality of GPT-4o-generated topics with established topic modeling techniques.

## Overview

This project implements a two-stage GPT-based topic modeling pipeline:
1. **Stage 1**: Extract topics from chunks of tweets using GPT-4o
2. **Stage 2**: Synthesize and refine topics from Stage 1 results

The approach is evaluated against multiple baseline methods including GSDMM, LDA, BERTopic variants, and their log-transformed versions.

## Files Structure

```
upload_code/
├── README.md                           # This file
├── run_experiments.ipynb              # Main experiment notebook
├── utils.py                           # Utility functions and prompts
├── gpt_topic_coherence.py            # Traditional topic coherence evaluation module
├── plots/                            # Generated visualization plots
│   ├── mean_scores_with_ci.png       # Semantic similarity scores comparison
│   └── mean_coherence_scores_with_ci.png  # Traditional topic coherence scores comparison
└── results/                          # Experimental results
    ├── baseline/                     # Baseline method results
    │   ├── *_topics.png             # Topic visualization plots
    │   ├── *_topn_df_5_random*.csv  # Top words for each topic
    │   └── score_df_5_random*.csv   # Traditional topic coherence scores
    └── gpt-4o-run*.pkl             # GPT experiment results
```

## Experimental Design

### Baseline Methods Evaluated

1. **Traditional Methods**:
   - GSDMM (Gibbs Sampling Dirichlet Multinomial Mixture)
   - LDA (Latent Dirichlet Allocation)

2. **BERTopic Variants**:
   - BERTopic with DistilRoBERTa embeddings
   - BERTopic with MiniLM embeddings  
   - BERTopic with MPNet embeddings

3. **Log-transformed Methods**:
   - Log-BDC-GSDMM
   - Log-BDC-LDA
   - Log-GSDMM
   - Log-LDA

### Evaluation Metrics

1. **Semantic Similarity**: Average pairwise similarity within topics using SentenceTransformer embeddings
2. **Topic Coherence**: C_V and U_mass coherence measures using Gensim


## Usage Instructions

### Prerequisites

```bash
pip install openai pandas tqdm numpy sentence-transformers gensim scipy matplotlib
```

### Running Experiments

1. **Set up API Key**: Replace `'YOUR_API_KEY'` in `run_experiments.ipynb` with your OpenAI API key

2. **Execute Notebook**: Run the cells in `run_experiments.ipynb` sequentially:
   - Load baseline data and preprocessed tweets
   - Run GPT-4o experiments with multiple random seeds
   - Evaluate baseline methods
   - Perform statistical comparisons
   - Generate visualization plots

3. **Customize Parameters**: Modify the following parameters as needed:
   - `topic_num`: Number of topics to extract (default: 5)
   - `chunk_size`: Size of tweet chunks for Stage 1 processing (default: 1000)

  ### 📖 Citation
If you find this work helpful, please consider citing our paper:

```bibtex
@article{he2025advanced,
  title={Advanced topic modeling with large language models: Analyzing social media content from dementia caregivers},
  author={He, Weiqing and Hou, Bojian and Zheng, Amy and Feng, Yanbo and Klein, Ari and O’Connor, Karen and Yang, Shu and Shang, Tianqi and Demiris, George and Gonzalez-Hernandez, Graciela and others},
  journal={Innovation in Aging},
  volume={9},
  number={Supplement\_1},
  pages={S38--S47},
  year={2025},
  publisher={Oxford University Press}
}
}
