from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_model(X_train, y_train):
    ''' 
    We are using Logistic Regression as our machine learning model
    to predict survival on the Titanic. Logistic Regression is a simple
    and effective algorithm for binary classification tasks, which is what
    we have here (survived vs. not survived). By setting max_iter=1000, 
    we ensure that the algorithm has enough iterations to converge to a solution,
    especially if the dataset is not linearly separable. This helps to improve 
    the chances of finding the optimal parameters for the model.

    A logistic regression model is a statistical model that in its basic
    form uses a logistic function to model a binary dependent variable. 
    In the context of the Titanic dataset, the logistic regression model
    will learn to predict the probability of survival based on the features
    provided in the training data. The model will output a value between 0 and 1,
    which can be interpreted as the probability of survival. By setting a threshold 
    (commonly 0.5), we can classify passengers as survivors or non-survivors based
    on their predicted probabilities.
    '''
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("\nModel Training Complete")
    os.makedirs("outputs/model", exist_ok=True)

    joblib.dump(model, "outputs/model/titanic_model.pkl")

    return model