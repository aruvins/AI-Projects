from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train import train_models
from src.evaluate import evaluate
from src.download_data import download_dataset

def main():
    download_dataset()
    
    df = load_data("data/housing.csv")

    preprocessor, X_train, X_test, y_train, y_test = preprocess_data(df)

    lr_model, rf_model = train_models(preprocessor, X_train, y_train)

    evaluate(lr_model, X_test, y_test, "Linear Regression")
    evaluate(rf_model, X_test, y_test, "Random Forest")

if __name__ == "__main__":
    main()