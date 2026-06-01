import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def encode(self, text):
        return torch.tensor([
            self.vocab.get(w, self.vocab["<unk>"])
            for w in text
        ])
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = self.texts[idx]
        encoded = [self.vocab.get(word, self.vocab["<unk>"]) for word in tokens]

        return(torch.tensor(encoded), torch.tensor(self.labels[idx]))
    
def collate_fn(batch):
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    texts = pad_sequence(
        texts,
        batch_first=True,
        padding_value=0
    )
    labels = torch.stack(labels)
    
    return texts, labels

def create_loader(texts, labels, vocab, batch_size=32):
    dataset = TextDataset(texts, labels, vocab)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )