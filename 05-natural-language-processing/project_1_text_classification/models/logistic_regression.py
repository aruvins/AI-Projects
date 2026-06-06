from sklearn.linear_model import LogisticRegression


class TextClassifier:
    def __init__(self):
        self.model = LogisticRegression(
            max_iter=1000,
            n_jobs=-1
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)