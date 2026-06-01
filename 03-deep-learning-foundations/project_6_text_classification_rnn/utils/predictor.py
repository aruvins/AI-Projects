import torch


def predict(sentence, model, vocab, device):
    model.eval()
    tokens = sentence.lower().split()

    encoded = [vocab.get(word, vocab["<unk>"]) for word in tokens]
    x = torch.tensor(encoded,dtype=torch.long).unsqueeze(0)
    x = x.to(device)

    with torch.no_grad():
        outputs = model(x)
        prediction = outputs.argmax(dim=1).item()
        probabilities = torch.softmax(outputs, dim=1)
        confidence = probabilities.max().item()
        
    return ("Positive" if prediction == 1 else "Negative", confidence)