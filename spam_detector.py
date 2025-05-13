import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def load_and_prepare_data(file):
    """loading dataset, droping unnecessary columns, and handle missing values"""
    ds = pd.read_csv(file)
    print(f"Original dataset shape: {ds.shape}")

    #droping 'Email No.'
    if 'Email No.' in ds.columns:
        ds = ds.drop(['Email No.'], axis=1)
        print("Dropped 'Email No.' column")

    #droping missing values
    if ds.isnull().values.any():
        missing_before = ds.isnull().sum().sum()
        ds = ds.dropna()
        missing_after = ds.isnull().sum().sum()
        print(f"Dropped rows with missing values: {missing_before - missing_after}")

    fs = ds.drop('Prediction', axis=1)
    lb = ds['Prediction']

    return fs, lb

def train_model(X_train, y_train, alpha=0.5):
    """training multinomial naive bayes model"""
    mod = MultinomialNB(alpha=alpha)
    mod.fit(X_train, y_train)
    print(f"Model trained with alpha={alpha}")
    return mod

def evaluate_model(mod, X_test, y_test):
    """evaluating the model"""
    predictions = mod.predict(X_test)

    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)

    print("\n=== Accuracy Score ===")
    print(f"{accuracy:.2f}")

    print("\n=== Classification Report ===")
    print(class_report)

    print("\n=== Confusion Matrix ===")
    print(conf_matrix)

    #ploting confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm')
    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

def main():
    #loading and preparing data
    fs, lb = load_and_prepare_data('emails.csv')

    #spliting data
    X_train, X_test, y_train, y_test = train_test_split(
        fs, lb, test_size=0.3, random_state=42, shuffle=True, stratify=lb
    )
    print(f"Training set size: {X_train.shape}, Testing set size: {X_test.shape}")

    #training the model
    spam_classifier = train_model(X_train, y_train, alpha=0.5)

    #evaluating the model
    evaluate_model(spam_classifier, X_test, y_test)

if __name__ == "__main__":
    main()
