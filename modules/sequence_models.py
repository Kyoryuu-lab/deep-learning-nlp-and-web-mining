"""
==============================================================================
Module:      sequence_models.py
Description: Recurrent Neural Network architectures (RNN, LSTM, GRU) for text
             classification and Seq2Seq Encoder-Decoder translation in PyTorch.
==============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import unicodedata
import re
from typing import List, Tuple, Dict


# --- 1. Seq2Seq Machine Translation (Encoder-Decoder GRU) ---

def normalize_string(s: str) -> str:
    s = unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("utf-8")
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s).lower().strip()
    return s


class TranslationVocabulary:
    def __init__(self):
        self.word2index = {"<SOS>": 0, "<EOS>": 1}
        self.index2word = {0: "<SOS>", 1: "<EOS>"}
        self.n_words = 2

    def add_sentence(self, sentence: str):
        for word in sentence.split():
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1


class EncoderGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_tensor, hidden):
        embedded = self.embedding(input_tensor).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden


class DecoderGRU(nn.Module):
    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden):
        embedded = self.embedding(input_tensor).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden


def run_seq2seq_demo(epochs: int = 300) -> str:
    """Trainiert ein kompaktes Seq2Seq-Modell (FR -> EN) und demonstriert die Übersetzung."""
    raw_pairs = [
        ("je suis heureux", "i am happy"),
        ("il fait beau", "the weather is nice"),
        ("nous apprenons l informatique", "we learn computer science")
    ]
    pairs = [(normalize_string(src), normalize_string(tgt)) for src, tgt in raw_pairs]

    src_vocab = TranslationVocabulary()
    tgt_vocab = TranslationVocabulary()
    for src, tgt in pairs:
        src_vocab.add_sentence(src)
        tgt_vocab.add_sentence(tgt)

    hidden_size = 128
    encoder = EncoderGRU(src_vocab.n_words, hidden_size)
    decoder = DecoderGRU(hidden_size, tgt_vocab.n_words)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)
    criterion = nn.NLLLoss()

    for epoch in range(1, epochs + 1):
        for src_sent, tgt_sent in pairs:
            src_tensor = torch.tensor([src_vocab.word2index[w] for w in src_sent.split()])
            tgt_tensor = torch.tensor([tgt_vocab.word2index[w] for w in tgt_sent.split()] + [1])

            encoder_hidden = torch.zeros(1, 1, hidden_size)
            optimizer.zero_grad()

            for word_idx in src_tensor:
                _, encoder_hidden = encoder(torch.tensor([word_idx]), encoder_hidden)

            decoder_input = torch.tensor([[0]])  # <SOS>
            decoder_hidden = encoder_hidden
            loss = 0

            for word_idx in tgt_tensor:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                loss += criterion(decoder_output, word_idx.unsqueeze(0))
                decoder_input = word_idx.unsqueeze(0)

            loss.backward()
            optimizer.step()

    # Evaluierungsfunktion
    test_sentence = pairs[0][0]
    with torch.no_grad():
        input_tensor = torch.tensor([src_vocab.word2index[w] for w in test_sentence.split()])
        encoder_hidden = torch.zeros(1, 1, hidden_size)
        for w in input_tensor:
            _, encoder_hidden = encoder(torch.tensor([w]), encoder_hidden)

        decoder_input = torch.tensor([[0]])
        decoder_hidden = encoder_hidden
        output_words = []

        for _ in range(10):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            top_idx = decoder_output.argmax().item()
            if top_idx == 1:  # <EOS>
                break
            output_words.append(tgt_vocab.index2word[top_idx])
            decoder_input = torch.tensor([[top_idx]])

    translated = " ".join(output_words)
    print(f"[*] Original (FR): \"{test_sentence}\"")
    print(f"[+] Uebersetzung (EN): \"{translated}\"")
    return translated


# --- 2. Recurrent Text Classifier (RNN / LSTM / GRU) ---

class RecurrentClassifier(nn.Module):
    def __init__(self, cell_type: str, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        cell = cell_type.upper()
        if cell == "LSTM":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        elif cell == "GRU":
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        if isinstance(self.rnn, nn.LSTM):
            _, (hidden, _) = self.rnn(embedded)
        else:
            _, hidden = self.rnn(embedded)
        return self.fc(hidden[-1])
