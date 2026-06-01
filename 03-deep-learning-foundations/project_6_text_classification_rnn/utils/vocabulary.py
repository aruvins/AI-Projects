from collections import Counter

def build_vocab(texts):
    counter = Counter()
    
    for text in texts:
        counter.update(text.lower().split())

    vocab = {
        "<pad>": 0,
        "<unk>": 1
    }

    for word, _ in counter.items():
        vocab[word] = len(vocab)

    return vocab