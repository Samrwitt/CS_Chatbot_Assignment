# NeuroSync: Hybrid Cognitive Next-Word Prediction & RAG Chatbot

NeuroSync is an advanced hybrid cognitive chatbot architecture that simulates human-like linguistic and reasoning processes. It combines a fast, reactive next-word predictor (**System 1**) with a slow, deliberate Retrieval-Augmented Generation (RAG) engine (**System 2**).

---

## 🧠 Core Architecture

The system operates on a dual-process cognitive model:

### 1. The Reactive Brain (System 1)
Designed for fast, intuitive pattern recognition and next-word prediction. It consists of two complementary architectures:
- **Connectionist RNN**: A PyTorch LSTM-based sequence predictor (`RNNWordPredictor`) trained on the domain-specific knowledge base text. It accepts context sequences of length 5 to predict the next token.
- **Statistical N-Grams**: A fallback/complementary linear-interpolated n-gram model (Unigram, Bigram, Trigram) trained on multiple corpora (Brown, Gutenberg, Reuters, etc.) and domain-specific texts. It uses linear interpolation weights (Trigram: 0.6, Bigram: 0.3, Unigram: 0.1) for smoothing.
- **WordNet Spreading Activation**: Simulates associative semantic retrieval by traversing the WordNet hierarchy (Synonyms, Hypernyms, and Hyponyms) to boost related candidate predictions.
- **Incremental User Learning**: Dynamically adapts to the user's linguistic style by learning from sent messages in real-time and persisting patterns to `user_memory.json`.

### 2. The Analytical Brain (System 2)
Provides high-level logical reasoning and contextual dialogue.
- **Semantic Retrieval**: A dense retriever (`SimpleRetriever`) using a SentenceTransformer (`all-MiniLM-L6-v2`) and cosine similarity to fetch the most relevant knowledge base chunks for a user query.
- **Contextual Augmentation**: Injects retrieved chunks into the prompt context for the local Large Language Model.
- **Ollama LLM Reasoning**: Delegates high-level response generation to a local Ollama instance (defaulting to `llama3`).
- **Graceful Fallback**: If the Ollama service is offline, System 1 automatically takes over to generate responses stochastically.

---

## ⚙️ Project Structure & Flow

- `run.py`: Entry point for launching the Flask web service (running on port 5005).
- `train_rnn.py`: CLI training script to extract text from `knowledge/`, build the vocabulary, split sequences into train/validation sets, train the LSTM network, and save weights to `app/rnn_model.pth`.
- `evaluate.py`: Evaluation suite that performs comparative analysis of all next-word predictors (RNN, Unigram, Bigram, Trigram, and Interpolated N-grams) on a held-out validation set.
- `app/predictor.py`: Implements the `AdvancedPredictor` class coordinating System 1 predictions and System 2 response generation.
- `app/retriever.py`: Implements SentenceTransformer semantic search for knowledge chunking and retrieval.
- `app/routes.py`: Flask API endpoints mapping the interface logic (`/predict`, `/chat`, and `/sync`).

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- **Ollama** (for System 2 reasoning)
  - Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
  - Pull Llama 3: `ollama pull llama3`

### Installation
1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: NLTK corpora (~100MB) will automatically download on the first execution.*

2. Train the Connectionist RNN:
   ```bash
   python3 train_rnn.py
   ```
   This will process PDFs and text files in `knowledge/`, split the data, train the LSTM, and save the model to `app/rnn_model.pth`.

3. Evaluate the models:
   ```bash
   python3 evaluate.py
   ```
   This generates a comparative performance table (Accuracies, Loss, Perplexity, and Latency) for all predictive models.

4. Run the backend service:
   ```bash
   python3 run.py
   ```

---

## 📊 Model Evaluation & Metrics

The system includes a CLI evaluation script (`evaluate.py`) that tests next-word prediction accuracy, cross-entropy loss, perplexity, and prediction speed on a held-out 10% validation set of the knowledge base.

### Performance Summary

| Model | Loss | Perplexity | Top-1 Acc | Top-3 Acc | Top-5 Acc | Avg Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Connectionist RNN (LSTM) | 3.8215 | 45.67 | 42.35% | 61.12% | 71.84% | 1.8410 ms |
| Unigram N-gram | 7.3688 | 1,585.77 | 6.05% | 13.22% | 19.45% | 0.1510 ms |
| Bigram N-gram | 6.2001 | 492.78 | 15.01% | 26.10% | 30.28% | 0.4832 ms |
| Trigram N-gram | 6.6268 | 755.03 | 12.92% | 21.47% | 26.44% | 0.1786 ms |
| Interpolated N-gram | 6.9508 | 1,044.02 | 16.92% | 27.11% | 32.19% | 1.5695 ms |

*Note: Latency measurements are averaged per token generation. RNN inference runs via PyTorch network forward pass, while N-grams leverage in-memory lookups.*
