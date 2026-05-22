from flask import Flask, request, jsonify, render_template
import requests
import random
import nltk
import json
from nltk.corpus import brown, webtext, nps_chat, gutenberg, reuters, wordnet, inaugural, movie_reviews, state_union, genesis, abc
from nltk import pos_tag, word_tokenize
from collections import defaultdict, Counter
import difflib
import re
import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Download necessary NLTK data
def download_nltk_data():
    resources = [
        'brown', 'webtext', 'nps_chat', 'gutenberg', 'reuters', 
        'punkt', 'averaged_perceptron_tagger', 'universal_tagset', 
        'wordnet', 'omw-1.4', 'inaugural', 'movie_reviews', 
        'state_union', 'genesis', 'abc'
    ]
    for res in resources:
        try:
            if '/' in res:
                nltk.data.find(res)
            else:
                nltk.data.find(f'corpora/{res}')
        except LookupError:
            nltk.download(res)

download_nltk_data()

class SimpleRetriever:
    def __init__(self, knowledge_dir='knowledge'):
        self.knowledge_dir = knowledge_dir
        self.chunks = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.ingest()

    def ingest(self):
        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir)
            return ""
        
        all_text = ""
        for filename in os.listdir(self.knowledge_dir):
            path = os.path.join(self.knowledge_dir, filename)
            text = ""
            if filename.endswith('.txt'):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        text = f.read()
                except Exception as e:
                    print(f"Error reading TXT {filename}: {e}")
            elif filename.endswith('.pdf'):
                try:
                    with open(path, 'rb') as f:
                        pdf = PyPDF2.PdfReader(f)
                        for page in pdf.pages:
                            text += page.extract_text() + "\n"
                except Exception as e:
                    print(f"Error reading PDF {filename}: {e}")
            
            if text:
                all_text += text + " "
                # Chunking by paragraph or fixed length
                file_chunks = [text[i:i+600] for i in range(0, len(text), 500)]
                self.chunks.extend(file_chunks)
        
        if self.chunks:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        
        return all_text

    def retrieve(self, query, top_k=2):
        if not self.chunks or self.tfidf_matrix is None:
            return ""
        
        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = [self.chunks[i] for i in top_indices if similarities[i] > 0.1]
            return "\n---\n".join(results)
        except:
            return ""

class AdvancedPredictor:
    def __init__(self, max_n=3, memory_file='user_memory.json'):
        self.max_n = max_n
        self.memory_file = memory_file
        self.models = [defaultdict(Counter) for _ in range(max_n + 1)]
        self.user_model = [defaultdict(Counter) for _ in range(max_n + 1)]
        self.weights = [0.1, 0.3, 0.6] # Weights for interpolation (unigram, bigram, trigram)
        self.retriever = SimpleRetriever()
        self.load_memory()
        
    def train(self, words):
        for n in range(1, self.max_n + 1):
            for i in range(len(words) - n + 1):
                context = tuple(words[i:i + n - 1])
                target = words[i + n - 1]
                self.models[n][context][target] += 1

    def learn_from_user(self, text):
        # Simulated "Neuroplasticity" - learning from the user's input
        words = re.findall(r'\b\w+\b', text.lower())
        for n in range(1, self.max_n + 1):
            for i in range(len(words) - n + 1):
                context = tuple(words[i:i + n - 1])
                target = words[i + n - 1]
                self.user_model[n][context][target] += 5 # Heavily weight user's own patterns
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
                    if n >= len(self.user_model): break
                    for context_str, targets in n_gram_model.items():
                        context = tuple(context_str.split()) if context_str else ()
                        self.user_model[n][context] = Counter(targets)
            print(f"Long-term memory loaded from {self.memory_file}")
        except Exception as e:
            print(f"Error loading memory: {e}")

    def predict_next(self, context_words):
        scores = Counter()
        
        # 1. Linear Interpolation Smoothing
        for n in range(1, min(self.max_n, len(context_words) + 2)):
            context = tuple(context_words[-(n-1):]) if n > 1 else ()
            combined = Counter(self.models[n][context])
            combined.update(self.user_model[n][context])
            
            if combined:
                total = sum(combined.values())
                weight = self.weights[n-1]
                for word, count in combined.items():
                    scores[word] += (count / total) * weight

        # 2. Semantic Spreading (WordNet) - Refined "Spreading Activation"
        # We simulate cognitive associations by traversing WordNet hierarchies
        top_candidates = scores.most_common(8)
        for word, score in top_candidates:
            related_words = set()
            for syn in wordnet.synsets(word):
                # Level 1: Synonyms
                for l in syn.lemmas():
                    if l.name().lower() != word.lower():
                        related_words.add(l.name().replace('_', ' '))
                
                # Level 2: Hypernyms (More general concepts)
                for hyper in syn.hypernyms():
                    for l in hyper.lemmas():
                        related_words.add(l.name().replace('_', ' '))
                
                # Level 3: Hyponyms (More specific concepts)
                for hypo in syn.hyponyms()[:3]: # Limit specifically to avoid explosion
                    for l in hypo.lemmas():
                        related_words.add(l.name().replace('_', ' '))

            for related in list(related_words)[:5]:
                # Inverse square law for activation decay
                scores[related] += score * 0.15 

        # 3. Diagnostic Data for UI Visualization
        diagnostic = {
            "pos": "Unknown",
            "confidence": 0,
            "process": "Statistical Brain"
        }
        
        if context_words:
            try:
                tags = pos_tag(context_words)
                diagnostic["pos"] = tags[-1][1]
            except: pass

        results = []
        for word, score in scores.most_common(10):
            results.append({"word": word, "score": round(score * 100, 1)})
            
        if results:
            diagnostic["confidence"] = results[0]["score"]

        return results[:5], diagnostic

    def generate_response(self, prompt_text, max_length=12, stream=False):
        # 1. RAG: Retrieve context from knowledge base
        context = self.retriever.retrieve(prompt_text)
        
        # 2. Attempt to use Ollama for High-Level Dialogue (Cognitive "System 2")
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
                timeout=20, # Increased for streaming
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

        # 2. Social Scripts Fallback
        prompt_text = prompt_text.lower().strip()
        greetings = ["hi", "hello", "hey", "greetings"]
        if any(g in prompt_text for g in greetings) and len(prompt_text.split()) < 3:
            return "Hello! I'm ready to chat. What's on your mind?"

        # 3. Anchored Stochastic N-gram Generation (Statistical "System 1")
        words = re.findall(r'\b\w+\b', prompt_text)
        stop_words = {'the', 'a', 'is', 'it', 'to', 'for', 'of', 'and', 'in', 'on', 'at'}
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        response = []
        context = keywords[-2:] if len(keywords) >= 2 else words[-2:]
        
        for i in range(max_length):
            preds, _ = self.predict_next(context)
            if not preds: break
            
            candidates = preds[:8]
            if i == 0:
                candidates = [p for p in candidates if p['word'] not in [',', '.', 'and', 'but', 'the', 'a']]
                if not candidates: candidates = preds[:8]

            weights = [p['score'] for p in candidates]
            total_w = sum(weights)
            if total_w == 0: break
            next_word = random.choices([p['word'] for p in candidates], weights=[w/total_w for w in weights])[0]
            
            if next_word in response[-3:]: continue 
            response.append(next_word)
            context = (context + [next_word])[-self.max_n+1:]
            
            if next_word in ['.', '!', '?']: break
                
        if not response:
            return "Tell me more about that."
            
        return " ".join(response).capitalize()

# Initialize and train the model
print("Training N-gram model with massive multi-corpus data... this may take a minute.")
train_words = []

# 1. Base corpora training
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

model = AdvancedPredictor(max_n=3)
model.train(train_words)

# 2. Domain-specific knowledge training (RAG integration into System 1)
knowledge_text = model.retriever.ingest()
if knowledge_text:
    k_words = re.findall(r'\b\w+\b|[.,!?]', knowledge_text.lower())
    # Heavily weight knowledge base patterns
    for n in range(1, model.max_n + 1):
        for i in range(len(k_words) - n + 1):
            context = tuple(k_words[i:i + n - 1])
            target = k_words[i + n - 1]
            model.models[n][context][target] += 10 # High weight for domain knowledge
            
print(f"Hybrid Cognitive Model trained successfully on {len(train_words)} base words + domain knowledge!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    history = data.get('history', [])
    
    if not text.strip() and not history:
        return jsonify({'predictions': []})
    
    # history learning removed from here to prevent redundant overweighting
    
    # Tokenize the input text
    words = re.findall(r'\b\w+\b|[.,!?]', text.lower())
    
    # If the text ends with a space, the user is starting a new word,
    # so we predict based on all words typed so far.
    # If it doesn't end with a space, the user is in the middle of typing a word.
    # In a full autocomplete, we would filter predictions that start with the partial word.
    # For this cognitive science project, we focus on predicting the *next* word given context.
    # We will assume they want the prediction for the next word only if they've typed a space,
    # OR we use the partial word to filter predictions.
    
    partial_word = ""
    if not text.endswith(' '):
        # User is in the middle of a word
        if words:
            partial_word = words.pop()
    
    predictions, diagnostic = model.predict_next(words)
    
    if partial_word:
        # Fuzzy Matching (Typo Correction)
        # 1. Try exact startswith first
        filtered = [p for p in predictions if p['word'].startswith(partial_word)]
        
        # 2. If no exact matches, use difflib for fuzzy matching
        if not filtered:
            all_words = [p['word'] for p in predictions]
            matches = difflib.get_close_matches(partial_word, all_words, n=3, cutoff=0.6)
            filtered = [p for p in predictions if p['word'] in matches]
        
        predictions = filtered
        
    return jsonify({
        'predictions': [p['word'] for p in predictions],
        'diagnostic': diagnostic,
        'scores': {p['word']: p['score'] for p in predictions}
    })

@app.route('/chat', methods=['POST'])
def chat():
    from flask import Response
    data = request.json
    message = data.get('message', '')
    
    if message:
        model.learn_from_user(message)
        
    def stream_response():
        # Retrieve context to determine if RAG or LLM-only
        context = model.retriever.retrieve(message)
        
        # Try to get a streaming response from Ollama
        gen = model.generate_response(message, stream=True)
        if hasattr(gen, '__iter__') and not isinstance(gen, (str, bytes)):
            process_name = "System 2 (RAG)" if context else "System 2 (Llama3)"
            yield f"data: {json.dumps({'process': process_name})}\n\n"
            for chunk in gen:
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        else:
            # Fallback to statistical brain (System 1)
            yield f"data: {json.dumps({'process': 'System 1 (Stochastic)'})}\n\n"
            resp = model.generate_response(message, stream=False)
            yield f"data: {json.dumps({'chunk': resp})}\n\n"
            
    return Response(stream_response(), mimetype='text/event-stream')

@app.route('/sync', methods=['POST'])
def sync_knowledge():
    # Re-ingest knowledge base
    knowledge_text = model.retriever.ingest()
    if knowledge_text:
        k_words = re.findall(r'\b\w+\b|[.,!?]', knowledge_text.lower())
        for n in range(1, model.max_n + 1):
            for i in range(len(k_words) - n + 1):
                context = tuple(k_words[i:i + n - 1])
                target = k_words[i + n - 1]
                model.models[n][context][target] += 10
    return jsonify({'status': 'success', 'message': 'Declarative memory synchronized.'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
