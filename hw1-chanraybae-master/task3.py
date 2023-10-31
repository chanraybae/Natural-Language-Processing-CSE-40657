import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import read_mono, Vocab

# Defining class of RNN Language Model
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2word = nn.Linear(hidden_dim, vocab_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        word_space = self.hidden2word(lstm_out.view(len(sentence), -1))
        word_scores = F.log_softmax(word_space, dim=1)
        return word_scores

# Helper function converting words to tensor
def prep_seq(seq, to_ix):
    idxs = [to_ix.get(w, to_ix["<UNK>"]) for w in seq]  # Default to <UNK> if word is not in vocab
    return torch.tensor(idxs, dtype=torch.long)

# Helper function to compute accuracy
def compute_accuracy(data, model, vocab):
    correct = 0
    total = 0
    with torch.no_grad():
        for sentence in data:
            inputs = prep_seq(sentence[:-1], vocab.word_to_num)
            targets = prep_seq(sentence[1:], vocab.word_to_num)
            log_probs = model(inputs)
            _, predicted = torch.max(log_probs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100 * correct / total

# Main code

vocab = Vocab()

# Training on small data
training_data_small = read_mono('data/small')
for sentence in training_data_small:
    vocab.update(sentence)

EMBEDDING_DIM = 128
HIDDEN_DIM = 128

model = RNN(len(vocab), EMBEDDING_DIM, HIDDEN_DIM)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(10):
    total_loss = 0
    for sentence in training_data_small:
        model.zero_grad()
        sentence_in = prep_seq(sentence[:-1], vocab.word_to_num)
        targets = prep_seq(sentence[1:], vocab.word_to_num)
        log_probs = model(sentence_in)
        loss = loss_function(log_probs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss}")

dev_data = read_mono('data/dev')
print(f"Small DM Accuracy: {compute_accuracy(dev_data, model, vocab):.2f}%")

# Training on large data
training_data_large = read_mono('data/large')
for sentence in training_data_large:
    vocab.update(sentence)

EMBEDDING_DIM = 512
HIDDEN_DIM = 512

model_large = RNN(len(vocab), EMBEDDING_DIM, HIDDEN_DIM)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model_large.parameters(), lr=0.1)

for epoch in range(10):
    total_loss = 0
    for sentence in training_data_large:
        model_large.zero_grad()
        sentence_in = prep_seq(sentence[:-1], vocab.word_to_num)
        targets = prep_seq(sentence[1:], vocab.word_to_num)
        log_probs = model_large(sentence_in)
        loss = loss_function(log_probs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss}")

print(f"Large DM Accuracy: {compute_accuracy(dev_data, model_large, vocab):.2f}%")
