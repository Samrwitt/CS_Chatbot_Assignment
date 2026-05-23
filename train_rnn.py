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
        
    # Split into train and validation sets (90% train, 10% val)
    split_idx = int(len(sequences) * 0.9)
    train_sequences = sequences[:split_idx]
    train_targets = targets[:split_idx]
    val_sequences = sequences[split_idx:]
    val_targets = targets[split_idx:]
    
    print(f"Split data: {len(train_sequences)} training sequences, {len(val_sequences)} validation sequences.")
    
    train_dataset = CognitiveDataset(train_sequences, train_targets)
    val_dataset = CognitiveDataset(val_sequences, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
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
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_seqs, batch_targets in train_loader:
            batch_seqs, batch_targets = batch_seqs.to(device), batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(batch_seqs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Validation Evaluation
        model.eval()
        val_loss = 0.0
        val_correct_top1 = 0
        val_correct_top5 = 0
        total_val_samples = len(val_dataset)
        
        with torch.no_grad():
            for batch_seqs, batch_targets in val_loader:
                batch_seqs, batch_targets = batch_seqs.to(device), batch_targets.to(device)
                outputs, _ = model(batch_seqs)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item() * batch_targets.size(0)
                
                # Top-1 accuracy
                _, top1_preds = outputs.topk(1, dim=1)
                val_correct_top1 += (top1_preds.squeeze(1) == batch_targets).sum().item()
                
                # Top-5 accuracy
                _, top5_preds = outputs.topk(min(5, vocab_size), dim=1)
                val_correct_top5 += sum([batch_targets[i].item() in top5_preds[i].tolist() for i in range(batch_targets.size(0))])
                
        val_loss /= total_val_samples
        val_perplexity = torch.exp(torch.tensor(val_loss)).item()
        val_acc_top1 = (val_correct_top1 / total_val_samples) * 100
        val_acc_top5 = (val_correct_top5 / total_val_samples) * 100
        
        train_loss_avg = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss_avg:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val PPL: {val_perplexity:.2f} | "
              f"Val Top-1 Acc: {val_acc_top1:.2f}% | Val Top-5 Acc: {val_acc_top5:.2f}%")
        
    # Save the trained model weights
    torch.save(model.state_dict(), 'app/rnn_model.pth')
    print("Saved model weights to app/rnn_model.pth")
    print("RNN training completed successfully!")

if __name__ == '__main__':
    train()
