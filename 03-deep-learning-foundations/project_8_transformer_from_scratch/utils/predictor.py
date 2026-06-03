import torch

from utils.dataset import encode_text


def predict_review(
    model,
    review,
    vocab,
    device,
    max_length=200
):
    """
    Predict sentiment for a single review.

    Args:
        model: trained Transformer
        review: raw text string
        vocab: vocabulary dictionary
        device: cpu/cuda/mps
        max_length: sequence length

    Returns:
        "Positive" or "Negative"
    """

    model.eval()

    encoded = encode_text(
        review,
        vocab,
        max_length
    )

    input_tensor = torch.tensor(
        encoded,
        dtype=torch.long
    ).unsqueeze(0)

    input_tensor = input_tensor.to(device)

    with torch.no_grad():

        outputs, _ = model(
            input_tensor
        )

        prediction = torch.argmax(
            outputs,
            dim=1
        ).item()

    if prediction == 1:
        return "Positive"

    return "Negative"