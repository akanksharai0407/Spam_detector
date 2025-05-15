# Spam Email Classification 🚫✉️

**A machine learning model that detects spam emails using a Naive Bayes classifier, implemented in Python with NLTK.**

---

## 📋 Dataset Description

- The dataset used is the **Email Spam Classification Dataset CSV** from Kaggle.  
- It contains **5172 emails** and **3002 features**.  
- The first column is the Email ID (dropped during preprocessing).  
- The last column is the label:  
  - `1` = spam  
  - `0` = not spam  
- The middle 3000 columns represent the most common words extracted from the emails.

---

## 🔧 How the Model Works

- The model is trained using a **Multinomial Naive Bayes classifier**.  
- It calculates the probability that an email belongs to the spam or non-spam class based on the words it contains.  
- Assumes each word contributes independently to the classification.  
- Uses NLTK for preprocessing emails: tokenization, stopword removal, and stemming.

---

## ⚙️ Project Structure

- **`spam_classifier.py`** — Main script that loads data, trains the model, and evaluates it.  
- **`emails.csv`** — The dataset used for training and testing.  
- **`confusion_matrix.png`** — Visualization of the model's confusion matrix.

---

## 📊 Evaluation Metrics

| Metric          | Description                                    | Result          |
|-----------------|------------------------------------------------|-----------------|
| **Accuracy**    | Percentage of correctly classified emails       | ~95%            |
| **F1 Score**    | Balance of precision and recall                  | 0.9236          |
| **Confusion Matrix** | True positives, false positives, etc.          | See below       |

---

## 🔍 Confusion Matrix

![output](https://github.com/user-attachments/assets/6dd8243b-f051-4817-a689-522b802cfc34)

---

## Output
<img width="601" alt="output" src="https://github.com/user-attachments/assets/366536e2-a082-4739-8c4e-7898f864d8b8" />


