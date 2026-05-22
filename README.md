#  NeuroSync

NeuroSync is a chatbot with a feature that suggests the next word based on the user's input, making typing faster and easier. It is useful for increased typing speed, reduced errors, and improved communication efficiency. It simulates human-like linguistic processes through a combination of **statistical associative memory (System 1)** and **Retrieval-Augmented Generation (RAG) intelligence (System 2)**.

---

##  Core Architecture

The system operates on a dual-process cognitive model:

### 1. The Statistical Brain (System 1)
Inspired by the human brain's fast, intuitive pattern recognition.
- **Interpolated N-Grams**: Uses a weighted combination of Unigram, Bigram, and Trigram models trained on over 1.6 million words from diverse corpora (Brown, Reuters, Gutenberg, etc.).
- **Smoothing**: Implements linear interpolation to handle sparse data and ensure smooth transitions between contexts.
- **Neuroplasticity (User Learning)**: The model dynamically adapts to your personal linguistic style in real-time. Every message you type strengthens specific neural pathways in the `user_model`.

### 2. Retrieval-Augmented Generation (RAG)
Provides the bot with a "Declarative Memory" or knowledge base.
- **Multi-Format Support**: Automatically ingests `.txt` and `.pdf` files from the `knowledge/` directory.
- **Semantic Retrieval**: Uses TF-IDF vectorization and Cosine Similarity to find the most relevant information for any user query.
- **Contextual Augmentation**: Injects retrieved knowledge directly into the high-level dialogue model (System 2) for more accurate and grounded responses.

### 3. Spreading Activation (Semantic Network)
Simulates how humans retrieve related concepts through associative memory.
- **WordNet Integration**: When predicting words, the model traverses the WordNet graph to find **Synonyms**, **Hypernyms** (general categories), and **Hyponyms** (specific instances).
- **Activation Decay**: Uses an inverse square law to simulate the natural fading of mental associations as they move further from the core concept.

### 4. High-Level Dialogue (System 2)
- **Ollama Integration**: If a local Ollama instance is running (defaulting to `llama3`), the system can delegate complex reasoning and conversational flow to a Large Language Model.
- **Graceful Fallback**: If System 2 is offline, the Statistical Brain takes over to maintain conversational continuity.

---

##  Features

- **Long-Term Memory Persistence**: All user-learned patterns are saved to `user_memory.json` and reloaded on startup.
- **Real-Time Diagnostics**:
  - **Linguistic Confidence**: Measures the statistical certainty of the next word prediction.
  - **Syntax Analysis**: Real-time Part-of-Speech (POS) tagging of user input.
  - **Process Monitor**: Identifies which part of the "brain" is currently generating the response.
- **Autocomplete & Ghost Text**: A premium UI experience providing predictive text completions (Tab/ArrowRight to accept).

---

##  Installation & Setup

### Prerequisites
- Python 3.10+
- **Ollama** (Required for System 2 / RAG features)
  - Install on Linux: `curl -fsSL https://ollama.com/install.sh | sh`
  - Pull the model: `ollama pull llama3`
  - Start the service: `ollama serve` (Usually starts automatically after install)
 
 ### How to Set it up 

**Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

**Run the Application**:
   ```bash
   python3 app.py
   ```
   *Note: On first run, NLTK data will automatically download (~100MB).*

---

##  User Interface

- **Chat Window**: Dynamic, responsive interface with glassmorphism aesthetics.
- **Ghost Text**: Visualizes the top-weighted prediction as you type.
- **Cognitive HUD**: Sidebar analytics showing the underlying mechanics of the AI's "thought" process.

---

##  Technical Implementation Details

- **Smoothing Weights**: Trigram (0.6), Bigram (0.3), Unigram (0.1).
- **Fuzzy Matching**: Uses `difflib` for typo correction in predictive text.
- **Training Source**: 200k words each from 10 major NLTK corpora.

