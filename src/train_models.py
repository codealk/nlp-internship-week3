import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

def train_and_evaluate():
    print("Loading mock dataset for classification...")
    # Simulated preprocessed data
    data = {
        'cleaned_text': [
            "internet not working router red light", 
            "how to update billing credit card",
            "what are your store hours",
            "wifi keeps dropping connection",
            "invoice amount is wrong double charged"
        ],
        'category': ['Tech_Support', 'Billing', 'General', 'Tech_Support', 'Billing']
    }
    df = pd.DataFrame(data)

    # Feature Extraction
    print("\nExtracting TF-IDF features...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['category']

    # Splitting data (Simulated)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model 1: Logistic Regression
    print("\n--- Training Logistic Regression ---")
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    print("Logistic Regression trained successfully.")

    # Model 2: Support Vector Machine (Linear SVC)
    print("\n--- Training Support Vector Machine (SVM) ---")
    svm_model = LinearSVC()
    svm_model.fit(X_train, y_train)
    print("SVM trained successfully.")

    print("\nNote: Use model.predict(X_test) and classification_report() for full metrics on a real dataset.")

if __name__ == "__main__":
    train_and_evaluate()    