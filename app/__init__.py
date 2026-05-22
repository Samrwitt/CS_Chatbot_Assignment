import nltk
from flask import Flask
from .predictor import AdvancedPredictor

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

def create_app():
    app = Flask(__name__)
    
    # Download NLTK corpora
    download_nltk_data()
    
    # Instantiate the AdvancedPredictor and bind to app
    app.model = AdvancedPredictor(max_n=3)
    
    # Register blueprints
    from .routes import main_bp
    app.register_blueprint(main_bp)
    
    return app
