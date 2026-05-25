from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def preprocess_data(df):
    # Select useful features
    df = df[
        [
            "Survived",
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
        ]
    ]

    # Handle missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


    # Encode categorical variables
    '''
    A label encoder is a simple way to convert categorical variables
    into numerical format. It assigns a unique integer to each category. 
    In this case, we are encoding the "Sex" and "Embarked" columns, which
    contain categorical data. By encoding these columns, we can use them
    as features in our machine learning model.
    '''
    encoder = LabelEncoder()

    df["Sex"] = encoder.fit_transform(df["Sex"])
    df["Embarked"] = encoder.fit_transform(df["Embarked"])

    # Features + target
    X = df.drop("Survived", axis=1)
    y = df["Survived"]


    # Train/test split
    '''
    The train_test_split function is used to split the dataset
    into a training set and a testing set. The training set is used
    to train the machine learning model, while the testing set is used
    to evaluate the model's performance on unseen data. 
    
    This is important to ensure that the model generalizes well and 
    does not overfit to the training data. By setting test_size=0.2, we
    are allocating 20% of the data for testing and 80% for training. 
    The random_state parameter is set to ensure reproducibility of the results.
    '''
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test