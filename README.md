# Deep Learning Projects — Carnegie Mellon University (11-785)

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![CMU](https://img.shields.io/badge/Carnegie_Mellon_University-CC0000?style=for-the-badge&logo=data:image/png;base64,&logoColor=white)]()

A collection of deep learning assignments completed as part of the **Introduction to Deep Learning (11-785)** course at Carnegie Mellon University. Each project covers a core area of modern deep learning — from building neural networks from scratch to sequence modelling and speech recognition — implemented in PyTorch.

---

## Projects Overview

| # | Project | Topic | Key Techniques |
|---|---------|-------|----------------|
| HW1 | Frame-Level Speech Classification | Feedforward Neural Networks | MLP, context windows, LibriSpeech |
| HW2 | Face Verification & Classification | Convolutional Neural Networks | ResNet, CNN, metric learning |
| HW3 | Automatic Speech Recognition | Sequence Modelling | RNN, LSTM, GRU, CTC decoding |
| HW4 | Advanced ASR | Attention & Sequence-to-Sequence | Transformer, attention mechanisms |

---

## Project Details

### HW1 — Frame-Level Speech Classification
**Topic:** Feedforward Neural Networks and Feature Engineering

The goal of this project was to classify phonemes from raw speech data using a multi-layer perceptron (MLP). The model takes MFCC (Mel-frequency cepstral coefficient) features extracted from the LibriSpeech dataset and predicts the phoneme label for each frame, using context windows to incorporate surrounding temporal information.

**Key concepts covered:**
- Building and training deep feedforward networks in PyTorch
- Feature engineering with context windows (temporal context around each frame)
- Dataset construction, DataLoader pipelines, and batch processing
- Hyperparameter tuning: learning rate, architecture depth, hidden layer size
- Model evaluation using accuracy score and Kaggle leaderboard submission

**Tech stack:** `PyTorch`, `NumPy`, `Pandas`, `sklearn`, `CUDA`

---

### HW2 — Face Verification and Classification
**Topic:** Convolutional Neural Networks and Metric Learning

This project tackled two tasks: face classification (identifying who a face belongs to across 7,000 classes) and face verification (determining whether two faces belong to the same person). A custom ResNet-34 architecture was implemented from scratch and trained with data augmentation strategies.

**Key concepts covered:**
- Implementing ResNet (BasicBlock, residual connections) from scratch
- CNN architecture design: convolutional layers, batch normalisation, stride/padding
- Data augmentation: random rotation, colour jitter, normalisation transforms
- Face verification using embedding similarity (cosine similarity)
- Transfer learning concepts and training on large-scale image datasets

**Tech stack:** `PyTorch`, `torchvision`, `PIL`, `ResNet-34`, `CUDA`

---

### HW3 — Automatic Speech Recognition (ASR)
**Topic:** Recurrent Neural Networks and Sequence Modelling

This project built an end-to-end automatic speech recognition system using sequence models. The task moved beyond frame-level classification to sequence-level transcription — mapping variable-length audio inputs to variable-length text outputs using CTC (Connectionist Temporal Classification) decoding.

**Key concepts covered:**
- Recurrent architectures: RNN, LSTM, GRU — understanding vanishing gradients and gating mechanisms
- CTC loss function and beam search decoding for sequence-to-sequence mapping
- Levenshtein distance as an evaluation metric for transcription quality
- Handling variable-length sequences with padding and packing
- Debugging shape errors in sequential models and CUDA memory management

**Tech stack:** `PyTorch`, `torchaudio`, `ctcdecode`, `Levenshtein`, `LSTM/GRU`, `CUDA`

---

### HW4 — Advanced Automatic Speech Recognition
**Topic:** Attention Mechanisms and Sequence-to-Sequence Models

An extension of the HW3 ASR system incorporating attention-based encoder-decoder architectures. This project explored how attention mechanisms allow the model to dynamically focus on relevant parts of the input sequence when generating each output token — a foundational concept behind modern transformer models.

**Key concepts covered:**
- Attention mechanisms: dot-product attention, scaled attention
- Encoder-decoder architecture for sequence-to-sequence tasks
- Comparison of RNN-based vs attention-based decoding strategies
- Model optimisation for quality vs latency tradeoffs
- Iterative experimentation and evaluation on ASR benchmarks

**Tech stack:** `PyTorch`, `LSTM`, `Attention`, `CTC`, `Beam Search`, `CUDA`

---

## Repository Structure

```
cmu-deep-learning/
│
├── hw1_speech_classification/
│   └── hw1p2_speech_classification.ipynb     # Frame-level phoneme classification (MLP)
│
├── hw2_face_recognition/
│   └── hw2p2_face_verification.ipynb          # Face classification & verification (ResNet CNN)
│
├── hw3_speech_recognition/
│   └── hw3p2_asr_sequence_models.ipynb        # ASR with RNN/LSTM/GRU + CTC decoding
│
├── hw4_attention_asr/
│   └── hw4p2_attention_asr.ipynb              # Advanced ASR with attention mechanisms
│
└── README.md
```

---

## Key Skills Demonstrated

- **Deep neural network design and training** — feedforward, convolutional, and recurrent architectures built from scratch in PyTorch
- **End-to-end ML pipelines** — data ingestion, preprocessing, model training, evaluation, and submission
- **Sequence modelling** — handling variable-length inputs with RNN, LSTM, GRU, and CTC decoding
- **Computer vision** — CNN architecture design with ResNet, data augmentation, and metric learning
- **Debugging and optimisation** — CUDA memory management, shape debugging, hyperparameter tuning, and performance iteration
- **Evaluation rigour** — Kaggle leaderboard submissions, Levenshtein distance, accuracy metrics

---

## Course Context

These projects were completed as part of **11-785: Introduction to Deep Learning** at **Carnegie Mellon University** during my Master of Science in Information Technology (magna cum laude, May 2023). The course is one of CMU's most rigorous ML courses, covering the theoretical foundations and practical implementation of modern deep learning systems.

---

## Note on Proprietary Work

Some of my most significant AI engineering projects — including **AnsysGPT** (a production RAG-based LLM chatbot shipped to Ansys customers, built on Azure OpenAI) and enterprise software solutions built for financial institutions at Cynergy Solutions — live in private company repositories and cannot be shared publicly. These projects are described in detail on my [LinkedIn profile](https://linkedin.com/in/iam-de-gee) and resume.

---

## Connect

- LinkedIn: [linkedin.com/in/iam-de-gee](https://linkedin.com/in/iam-de-gee)
- Email: adebayodeji147@gmail.com
