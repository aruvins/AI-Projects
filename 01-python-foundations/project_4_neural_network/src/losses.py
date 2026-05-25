import numpy as np


class CrossEntropyLoss:
    def forward(self, predictions, y_true):
        '''
        Computes the cross-entropy loss between predictions and true labels.
        
        The cross-entropy loss is defined as:
        L = -1/N * sum(y_true * log(predictions))
        
        where N is the number of samples, 
        y_true is the true label (one-hot encoded or class indices), 
        and predictions are the predicted probabilities for each class.

        To ensure numerical stability, the predictions are clipped to a small range before taking the logarithm

        Parameters:
        - predictions: The predicted probabilities for each class (output of the softmax activation).
        - y_true: The true labels for the samples (can be class indices or one-hot encoded).
        '''

        samples = len(predictions)

        predictions_clipped = np.clip(
            predictions,
            1e-7, 
            1 - 1e-7,
        ) # Clip to prevent log(0)

        correct_confidences = predictions_clipped[
            range(samples),
            y_true,
        ] # Get the predicted probabilities for the correct classes

        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)

    def backward(self, predictions, y_true):
        ''' 
        Computes the gradient of the cross-entropy loss with respect to the predictions.
        The gradient is calculated as:
        grad = predictions - y_true

        For a batch of samples, the gradient is averaged over the number of samples to ensure that the scale of the gradients remains consistent regardless of batch size.
        Parameters:
        - predictions: The predicted probabilities for each class (output of the softmax activation).
        - y_true: The true labels for the samples (can be class indices or one-hot encoded).

        '''
        samples = len(predictions)
        grad = predictions.copy() 
        
        grad[range(samples), y_true] -= 1 # Subtract 1 from the predicted probabilities of the correct classes
        grad = grad / samples # Average the gradients over the batch

        return grad