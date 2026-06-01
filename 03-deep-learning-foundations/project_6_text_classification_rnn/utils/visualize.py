import matplotlib.pyplot as plt

def plot_metrics(model, losses,accuracies):
    # Training Loss
    plt.figure()
    plt.plot(losses)
    plt.title(f"Training Loss: {model}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"outputs/loss_curve_{model}.png")
    plt.show()

    # Training Accuracy
    plt.figure()
    plt.plot(accuracies)
    plt.title(f"Training Accuracy: {model}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(f"outputs/accuracy_curve_{model}.png")
    plt.show()


def compare_models(rnn_losses, lstm_losses, rnn_acc, lstm_acc):
    # Training Loss Comparison
    plt.figure()
    plt.plot(rnn_losses,label="RNN")
    plt.plot(lstm_losses, label="LSTM")
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("outputs/loss_comparison.png")
    plt.show()

    # Training Accuracy Comparison
    plt.figure()
    plt.plot(rnn_acc, label="RNN") 
    plt.plot(lstm_acc, label="LSTM")
    plt.title("Training Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("outputs/accuracy_comparison.png")
    plt.show()