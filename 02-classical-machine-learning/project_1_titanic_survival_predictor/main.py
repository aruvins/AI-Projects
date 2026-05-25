from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.download_data import download_dataset

def main():
    # Download data
    download_dataset()
    
    # Load data
    df = load_data("data/titanic.csv")

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()