import os
import PyPDF2
import torch
from sentence_transformers import SentenceTransformer, util

class SimpleRetriever:
    def __init__(self, knowledge_dir='knowledge', model_name='all-MiniLM-L6-v2'):
        self.knowledge_dir = knowledge_dir
        self.chunks = []
        self.model = SentenceTransformer(model_name)
        self.chunk_embeddings = None
        self.ingest()

    def ingest(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        actual_knowledge_dir = os.path.join(project_root, self.knowledge_dir)
        
        if not os.path.exists(actual_knowledge_dir):
            os.makedirs(actual_knowledge_dir)
            return ""
        
        all_text = ""
        for filename in os.listdir(actual_knowledge_dir):
            path = os.path.join(actual_knowledge_dir, filename)
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
            self.chunk_embeddings = self.model.encode(self.chunks, convert_to_tensor=True)
        
        return all_text

    def retrieve(self, query, top_k=2):
        if not self.chunks or self.chunk_embeddings is None:
            return ""
        
        try:
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, self.chunk_embeddings)[0]
            top_results = torch.topk(cos_scores, k=min(top_k, len(self.chunks)))
            
            results = []
            for score, idx in zip(top_results[0], top_results[1]):
                if score.item() > 0.35:
                    results.append(self.chunks[idx.item()])
            return "\n---\n".join(results)
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return ""
