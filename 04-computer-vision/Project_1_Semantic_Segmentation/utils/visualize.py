import matplotlib.pyplot as plt


def show_prediction(image, mask, prediction):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(image)
    ax[0].set_title("Image")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Ground Truth")

    ax[2].imshow(prediction, cmap="gray")
    ax[2].set_title("Prediction")

    plt.show()