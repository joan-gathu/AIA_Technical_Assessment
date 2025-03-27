import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from document_processing_pipeline import DataPreprocessor, EmbeddingGenerator


class SentimentClassifier:
    """
    Sentiment analysis classifier using logistic regression and embeddings.
    """

    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.embedder = EmbeddingGenerator()
        self.model = LogisticRegression(max_iter=1000)
        self.label_encoder = LabelEncoder()

    def preprocess_data(self, df):
        """preprocess data, generates embeddings, and encodes labels."""
        df.dropna(subset=["review_text", "sentiment"], inplace=True)
        df["review_text"] = df["review_text"].apply(self.preprocessor.clean_text)
        print("Generating embeddings... (this may take some time)")
        df["embeddings"] = df["review_text"].apply(self.embedder.generate_embedding)

        X = np.vstack(df["embeddings"].values)
        y = self.label_encoder.fit_transform(df["sentiment"])

        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def train(self, X_train, y_train):
        """Trains the logistic regression model."""
        print("Training Logistic Regression model...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Evaluates the trained model."""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)

        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:\n", report)

        return accuracy, report

if __name__ == "__main__":
    # Use absolute path for better file handling
    file_path = os.path.join(os.getcwd(), "data", "product_reviews.csv")

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        classifier = SentimentClassifier()

        # preprocess data
        X_train, X_test, y_train, y_test = classifier.preprocess_data(df)

        # Train the model
        classifier.train(X_train, y_train)

        # Evaluate the model
        classifier.evaluate(X_test, y_test)
    else:
        print(f"Error: File not found at {file_path}")
