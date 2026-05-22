import os
import re
import json
import PyPDF2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class CognitiveDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class RNNWordPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        out, hidden = self.lstm(embeds, hidden)
        out = self.fc(out[:, -1, :])  # Predict the next word after the sequence
        return out, hidden

def extract_knowledge_text(knowledge_dir='knowledge'):
    print("Extracting text from knowledge base...")
    all_text = ""
    if not os.path.exists(knowledge_dir):
        print(f"Error: {knowledge_dir} directory not found.")
        return ""
        
    for filename in os.listdir(knowledge_dir):
        path = os.path.join(knowledge_dir, filename)
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
            print(f" - Loaded {filename} ({len(text)} characters)")
            all_text += text + " "
    return all_text

def train():
    raw_text = extract_knowledge_text()
    if not raw_text:
        print("No text found in knowledge base. Training aborted.")
        return
        
    # Clean and tokenize
    print("Tokenizing text...")
    tokens = re.findall(r'\b\w+\b|[.,!?]', raw_text.lower())
    print(f"Total tokens: {len(tokens)}")
    
    # Build vocabulary
    word_counts = {}
    for token in tokens:
        word_counts[token] = word_counts.get(token, 0) + 1
        
    # Filter very rare words to keep model size small and fast
    vocab = ["<PAD>", "<UNK>"] + [word for word, count in word_counts.items() if count >= 1]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for idx, word in enumerate(vocab)}
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Save vocabulary
    with open('app/vocab.json', 'w', encoding='utf-8') as f:
        json.dump({
            "word_to_idx": word_to_idx,
            "idx_to_word": idx_to_word
        }, f, indent=4)
    print("Saved vocabulary to app/vocab.json")
    
    # Create sequences (Context length = 5)
    seq_length = 5
    sequences = []
    targets = []
    
    for i in range(len(tokens) - seq_length):
        seq = tokens[i : i + seq_length]
        target = tokens[i + seq_length]
        
        seq_idx = [word_to_idx.get(w, word_to_idx["<UNK>"]) for w in seq]
        target_idx = word_to_idx.get(target, word_to_idx["<UNK>"])
        
        sequences.append(seq_idx)
        targets.append(target_idx)
        
    print(f"Total training sequences: {len(sequences)}")
    
    # Sub-sample sequences if too large, to keep local training extremely fast on CPU
    max_train_sequences = 30000
    if len(sequences) > max_train_sequences:
        sequences = sequences[:max_train_sequences]
        targets = targets[:max_train_sequences]
        print(f"Sub-sampled to {len(sequences)} sequences for fast local CPU training.")
        
    dataset = CognitiveDataset(sequences, targets)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Model parameters
    vocab_size = len(vocab)
    embedding_dim = 64
    hidden_dim = 128
    epochs = 15
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = RNNWordPredictor(vocab_size, embedding_dim, hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_seqs, batch_targets in dataloader:
            batch_seqs, batch_targets = batch_seqs.to(device), batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(batch_seqs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")
        
    # Save the trained model weights
    torch.save(model.state_dict(), 'app/rnn_model.pth')
    print("Saved model weights to app/rnn_model.pth")
    print("RNN training completed successfully!")

if __name__ == '__main__':
    train()
