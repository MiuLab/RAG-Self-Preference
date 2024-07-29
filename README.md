# RAG Self Preference

## Abstract
In the rapidly evolving field of information retrieval, the retrieval-augmented generation (RAG) framework offers a revolutionary approach to accessing knowledge. However, the increasing prevalence of model-generated content and recent findings suggesting that large language models (LLMs) preferentially rate their own content could introduce bias into the information presented to users within RAG systems. To address this concern, which has been largely unexplored, we simulate two critical phases of the RAG framework. Initially, we engage `GPT-3.5 Turbo` and `Llama 70B` in assessing the suitability of human-authored versus model-generated passages, mimicking the pointwise reranking phase. Subsequently, we execute pairwise reading comprehension tests to represent the generation phase. Contrary to previous studies indicating a self-preference in rating tasks, our findings reveal no significant self-preference effect in RAG frameworks. Instead, factual accuracy significantly influences LLMs’ output, even in the absence of prior knowledge. Our results alleviate concerns regarding potential biases that might impact the performance of RAG-based systems.

## Experiments
[TODO] Overview of Experiments
[TODO] Dataset Reference and Introduction
![overview](D:\Lily\college\CSIE\112-2\research\RAG-Self-Preference\graphs\overview.png)

## How to Execute
[TODO] Clone and Environment Setup
[TODO] API replacement
To obtain the tables and results for the generation phase, simply run:
```bash
./Generation\ Phase/run.sh
```
To obtain the tables and results for the pointwise reranking phase, simply run:
```bash
./Pointwise\ Reranking\ Phase/run.sh
```