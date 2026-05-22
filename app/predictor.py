import os
import re
import random
import json
import requests
import nltk
from collections import defaultdict, Counter
from nltk.corpus import brown, webtext, nps_chat, gutenberg, reuters, wordnet, inaugural, movie_reviews, state_union, genesis, abc
from nltk import pos_tag
import torch
import torch.nn as nn

from .retriever import SimpleRetriever

class RNNWordPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        out, hidden = self.lstm(embeds, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

class AdvancedPredictor:
    def __init__(self, max_n=3, memory_file='user_memory.json'):
        self.max_n = max_n
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.memory_file = os.path.join(project_root, memory_file)
        self.models = [defaultdict(Counter) for _ in range(max_n + 1)]
        self.user_model = [defaultdict(Counter) for _ in range(max_n + 1)]
        self.weights = [0.1, 0.3, 0.6]  # Weights for interpolation (unigram, bigram, trigram)
        self.retriever = SimpleRetriever()
        
        self.vocab = set()
        
        # Train statistical model and load user memory
        self.train_model()
        self.load_memory()
        
        # Check for trained RNN model (Connectionist Brain)
        self.use_rnn = False
        app_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(app_dir, 'rnn_model.pth')
        vocab_path = os.path.join(app_dir, 'vocab.json')
        
        if os.path.exists(model_path) and os.path.exists(vocab_path):
            try:
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    vocab_data = json.load(f)
                    self.word_to_idx = vocab_data["word_to_idx"]
                    self.idx_to_word = {int(k): v for k, v in vocab_data["idx_to_word"].items()}
                
                vocab_size = len(self.word_to_idx)
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                self.rnn_model = RNNWordPredictor(vocab_size, embedding_dim=64, hidden_dim=128).to(self.device)
                self.rnn_model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.rnn_model.eval()
                self.use_rnn = True
                print("Connectionist RNN next-word prediction model loaded successfully!")
            except Exception as e:
                print(f"Error loading RNN model: {e}. Falling back to statistical N-grams.")
        else:
            print("RNN weights/vocab not found. Run 'python train_rnn.py' to train. Using N-gram fallback.")

    def train_model(self):
        print("Training statistical N-gram model with multi-corpus data...")
        train_words = []
        corpora_list = [
            brown, webtext, nps_chat, gutenberg, reuters, 
            inaugural, movie_reviews, state_union, genesis, abc
        ]

        for corpus in corpora_list:
            try:
                words = corpus.words()[:200000]
                train_words.extend([word.lower() for word in words if word.isalpha() or word in ['.', ',', '!', '?']])
            except Exception as e:
                print(f"Skipping corpus due to error: {e}")

        # Build vocabulary from training corpora
        self.vocab = set(w for w in train_words if w.isalpha())

        # Train statistical n-gram model
        for n in range(1, self.max_n + 1):
            for i in range(len(train_words) - n + 1):
                context = tuple(train_words[i:i + n - 1])
                target = train_words[i + n - 1]
                self.models[n][context][target] += 1

        # Train with domain-specific knowledge (RAG integration into System 1)
        knowledge_text = self.retriever.ingest()
        if knowledge_text:
            k_words = re.findall(r'\b\w+\b|[.,!?]', knowledge_text.lower())
            for n in range(1, self.max_n + 1):
                for i in range(len(k_words) - n + 1):
                    context = tuple(k_words[i:i + n - 1])
                    target = k_words[i + n - 1]
                    self.models[n][context][target] += 10
            
            # Add domain words to vocabulary
            for w in k_words:
                if w.isalpha():
                    self.vocab.add(w)
                    
        print(f"Statistical model trained successfully on {len(train_words)} base words! Vocab size: {len(self.vocab)}")

    def train(self, words):
        for n in range(1, self.max_n + 1):
            for i in range(len(words) - n + 1):
                context = tuple(words[i:i + n - 1])
                target = words[i + n - 1]
                self.models[n][context][target] += 1

    def learn_from_user(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        for n in range(1, self.max_n + 1):
            for i in range(len(words) - n + 1):
                context = tuple(words[i:i + n - 1])
                target = words[i + n - 1]
                self.user_model[n][context][target] += 5
        self.save_memory()

    def save_memory(self):
        try:
            serializable_model = []
            for n in range(len(self.user_model)):
                n_gram_model = {}
                for context, targets in self.user_model[n].items():
                    context_str = " ".join(context)
                    n_gram_model[context_str] = dict(targets)
                serializable_model.append(n_gram_model)
            
            with open(self.memory_file, 'w') as f:
                json.dump(serializable_model, f)
        except Exception as e:
            print(f"Error saving memory: {e}")

    def load_memory(self):
        if not os.path.exists(self.memory_file):
            return
        try:
            with open(self.memory_file, 'r') as f:
                data = json.load(f)
                for n, n_gram_model in enumerate(data):
                    if n >= len(self.user_model): 
                        break
                    for context_str, targets in n_gram_model.items():
                        context = tuple(context_str.split()) if context_str else ()
                        self.user_model[n][context] = Counter(targets)
            print(f"Long-term memory loaded from {self.memory_file}")
        except Exception as e:
            print(f"Error loading memory: {e}")

    def predict_next(self, context_words):
        scores = Counter()
        
        # 1. Prediction with RNN model if available
        if self.use_rnn:
            try:
                # Use context words (context window size = 5)
                context_idx = []
                for w in context_words[-5:]:
                    context_idx.append(self.word_to_idx.get(w, self.word_to_idx["<UNK>"]))
                
                # Pad sequence if too short
                if len(context_idx) < 5:
                    context_idx = [self.word_to_idx["<PAD>"]] * (5 - len(context_idx)) + context_idx
                    
                context_tensor = torch.tensor([context_idx], dtype=torch.long).to(self.device)
                
                with torch.no_grad():
                    logits, _ = self.rnn_model(context_tensor)
                    probabilities = torch.softmax(logits, dim=-1).flatten()
                    
                # Get top predictions
                top_probs, top_indices = torch.topk(probabilities, k=min(15, len(self.word_to_idx)))
                
                for prob, idx in zip(top_probs, top_indices):
                    word = self.idx_to_word.get(idx.item())
                    if word and word not in ["<PAD>", "<UNK>"]:
                        scores[word] = prob.item()
            except Exception as e:
                print(f"RNN prediction error: {e}. Falling back to N-grams.")
                self.use_rnn = False
                
        # 2. Fallback to Linear Interpolation N-gram Smoothing
        if not self.use_rnn:
            for n in range(1, min(self.max_n, len(context_words) + 2)):
                context = tuple(context_words[-(n-1):]) if n > 1 else ()
                combined = Counter(self.models[n][context])
                combined.update(self.user_model[n][context])
                
                if combined:
                    total = sum(combined.values())
                    weight = self.weights[n-1]
                    for word, count in combined.items():
                        scores[word] += (count / total) * weight

        # Repetition Penalty: Penalize words based on distance in recent context
        if context_words:
            for word in list(scores.keys()):
                try:
                    distance = list(reversed(context_words)).index(word) + 1
                    if distance <= 12:
                        penalty_factor = 0.1 + 0.9 * ((distance - 1) / 11)
                        scores[word] *= penalty_factor
                except ValueError:
                    pass

        # 3. Semantic Spreading (WordNet)
        top_candidates = scores.most_common(8)
        for word, score in top_candidates:
            if not word.isalpha():
                continue
            related_words = set()
            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    name = l.name().lower().replace('_', ' ')
                    if name != word.lower() and (name in self.vocab or (self.use_rnn and name in self.word_to_idx)):
                        related_words.add(name)
                
                for hyper in syn.hypernyms():
                    for l in hyper.lemmas():
                        name = l.name().lower().replace('_', ' ')
                        if (name in self.vocab or (self.use_rnn and name in self.word_to_idx)):
                            related_words.add(name)
                
                for hypo in syn.hyponyms()[:3]: 
                    for l in hypo.lemmas():
                        name = l.name().lower().replace('_', ' ')
                        if (name in self.vocab or (self.use_rnn and name in self.word_to_idx)):
                            related_words.add(name)

            for related in list(related_words)[:5]:
                scores[related] += score * 0.15 

        # Diagnostic Data for UI Visualization
        diagnostic = {
            "pos": "Unknown",
            "confidence": 0,
            "process": "Connectionist RNN Brain" if self.use_rnn else "Statistical Brain"
        }
        
        if context_words:
            try:
                tags = pos_tag(context_words)
                diagnostic["pos"] = tags[-1][1]
            except: 
                pass

        results = []
        for word, score in scores.most_common(10):
            # Scale scores to 0-100 percentage for display
            display_score = score * 100
            results.append({"word": word, "score": round(display_score, 1)})
            
        if results:
            diagnostic["confidence"] = results[0]["score"]

        return results[:5], diagnostic

    def generate_response(self, prompt_text, max_length=12, stream=False):
        context = self.retriever.retrieve(prompt_text)
        
        try:
            augmented_prompt = f"You are a helpful cognitive science chatbot. Use the following context if relevant: \n\n{context}\n\nUser said: {prompt_text}"
            if not context:
                augmented_prompt = f"You are a helpful cognitive science chatbot. Be brief. User said: {prompt_text}"
                
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3",
                    "prompt": augmented_prompt,
                    "stream": stream
                },
                timeout=20,
                stream=stream
            )
            
            if stream:
                def generate():
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            yield data.get('response', '')
                            if data.get('done'):
                                break
                return generate()
            
            if response.status_code == 200:
                return response.json().get('response', '').strip()
        except Exception as e:
            print(f"Ollama Error: {e}")
            pass

        # Social Scripts Fallback
        prompt_text = prompt_text.lower().strip()
        greetings = ["hi", "hello", "hey", "greetings"]
        if any(g in prompt_text for g in greetings) and len(prompt_text.split()) < 3:
            return "Hello! I'm ready to chat. What's on your mind?"

        # Fallback to Statistical System 1 Generation with loop prevention
        words = re.findall(r'\b\w+\b', prompt_text)
        stop_words = {'the', 'a', 'is', 'it', 'to', 'for', 'of', 'and', 'in', 'on', 'at'}
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        response = []
        context = keywords[-2:] if len(keywords) >= 2 else words[-2:]
        generated_bigrams = set()
        
        for i in range(max_length):
            preds, _ = self.predict_next(context)
            if not preds: 
                break
            
            filtered_candidates = []
            last_word = response[-1] if response else (context[-1] if context else None)
            
            for p in preds:
                cand_word = p['word']
                if last_word and cand_word.isalpha() and last_word.isalpha():
                    bigram = (last_word.lower(), cand_word.lower())
                    if bigram in generated_bigrams:
                        continue
                filtered_candidates.append(p)
                
            candidates = filtered_candidates[:8]
            if not candidates:
                candidates = preds[:8]
                
            if i == 0:
                candidates = [p for p in candidates if p['word'] not in [',', '.', 'and', 'but', 'the', 'a']]
                if not candidates: 
                    candidates = preds[:8]

            weights = [p['score'] for p in candidates]
            total_w = sum(weights)
            if total_w == 0: 
                break
            next_word = random.choices([p['word'] for p in candidates], weights=[w/total_w for w in weights])[0]
            
            if last_word and next_word.isalpha() and last_word.isalpha():
                generated_bigrams.add((last_word.lower(), next_word.lower()))
                
            response.append(next_word)
            context = (context + [next_word])[-self.max_n+1:]
            
            if next_word in ['.', '!', '?']: 
                break
                
        if not response:
            return "Tell me more about that."
            
        return " ".join(response).capitalize()
