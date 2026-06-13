import os
import subprocess


MODEL_PATH = "outputs/chatbot_model.pth"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def train_model():

    print("\n" + "=" * 60)
    print("TRAINING CHATBOT MODEL")
    print("=" * 60)

    subprocess.run(
        ["python", "train.py"],
        cwd=BASE_DIR,
        check=True
    )


def launch_chatbot():

    print("\n" + "=" * 60)
    print("LAUNCHING CHATBOT")
    print("=" * 60)

    subprocess.run(
        ["python", "chatbot.py"],
        cwd=BASE_DIR,
        check=True
    )


def main():

    print("\nCHATBOT BASICS PROJECT\n")

    if not os.path.exists(MODEL_PATH):

        print(
            "No trained model found."
        )

        train_model()

    else:

        print(
            "Existing trained model found."
        )

        retrain = input(
            "Retrain model? (y/n): "
        ).lower()

        if retrain == "y":

            train_model()

    launch_chatbot()


if __name__ == "__main__":
    main()