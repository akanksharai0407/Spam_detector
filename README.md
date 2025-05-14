# Spam_detector

Project Summary:

This project is about building a simple spam email detector using the Naive Bayes algorithm. The goal is to figure out whether an email is spam or not, based on the words it contains. The model was trained using a dataset from Kaggle and was implemented in Python. I also used NLTK for some text preprocessing.

Dataset Info:

The dataset I used is the Email Spam Classification Dataset from Kaggle. It includes:

Around 5000+ emails

3000 columns representing the most common words in the emails

A final column that marks whether an email is spam (1) or not (0)

Each row is basically an email, represented by how often each word appears.

Model Used:

I used the Multinomial Naive Bayes classifier, which is good for text data. The idea is to look at how often words appear in spam emails vs. normal ones, and then use that to guess whether a new email is spam.

Steps I followed:

Loaded the dataset and split it into training and testing sets

Trained the Naive Bayes model on the training data

Tested it on the test data to see how well it performs

Text Preprocessing (with NLTK):

To test the model with new input emails, I used NLTK to:

Tokenize the input (split into words)

Remove stopwords (common words like “the”, “and”)

Do stemming (reduce words like “running” to “run”)

This helps clean the input and match it better with the features in the original dataset.

Results & Evaluation:

I used the following metrics to evaluate the model:

Accuracy: about 95%

F1 Score: around 0.95, which balances precision and recall

Confusion Matrix: I created a heatmap to show correct and incorrect predictions

Conclusion
Even though this is a pretty basic approach, it works well for detecting spam based on email content. The dataset didn’t include sender domain info, so I focused only on the words used in the emails.
