from flask import Blueprint, request, jsonify, render_template, Response, current_app
import re
import difflib
import json

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/predict', methods=['POST'])
def predict():
    model = current_app.model
    data = request.json
    text = data.get('text', '')
    history = data.get('history', [])
    
    if not text.strip() and not history:
        return jsonify({'predictions': []})
        
    # Tokenize the input text
    words = re.findall(r'\b\w+\b|[.,!?]', text.lower())
    
    partial_word = ""
    if not text.endswith(' '):
        if words:
            partial_word = words.pop()
    
    predictions, diagnostic = model.predict_next(words)
    
    if partial_word:
        # Fuzzy Matching (Typo Correction)
        filtered = [p for p in predictions if p['word'].startswith(partial_word)]
        
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

@main_bp.route('/chat', methods=['POST'])
def chat():
    model = current_app.model
    data = request.json
    message = data.get('message', '')
    
    if message:
        model.learn_from_user(message)
        
    def stream_response():
        context = model.retriever.retrieve(message)
        gen = model.generate_response(message, stream=True)
        if hasattr(gen, '__iter__') and not isinstance(gen, (str, bytes)):
            process_name = "System 2 (RAG)" if context else "System 2 (Llama3)"
            yield f"data: {json.dumps({'process': process_name})}\n\n"
            for chunk in gen:
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        else:
            yield f"data: {json.dumps({'process': 'System 1 (Stochastic)'})}\n\n"
            resp = model.generate_response(message, stream=False)
            yield f"data: {json.dumps({'chunk': resp})}\n\n"
            
    return Response(stream_response(), mimetype='text/event-stream')

@main_bp.route('/sync', methods=['POST'])
def sync_knowledge():
    model = current_app.model
    knowledge_text = model.retriever.ingest()
    if knowledge_text:
        k_words = re.findall(r'\b\w+\b|[.,!?]', knowledge_text.lower())
        for n in range(1, model.max_n + 1):
            for i in range(len(k_words) - n + 1):
                context = tuple(k_words[i:i + n - 1])
                target = k_words[i + n - 1]
                model.models[n][context][target] += 10
    return jsonify({'status': 'success', 'message': 'Declarative memory synchronized.'})
