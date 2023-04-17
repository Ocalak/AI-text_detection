import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression








# Load dataset of AI-generated and human-written text
df = pd.read_csv("/Users/ocalkaptan/Desktop/text-detection/df.csv")
print(df["text1"])
"""
# Split dataset into training and testing sets
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])

# Define the labels for AI-generated (1) and human-written (0) text
y_train = train_df["label"]
y_test = test_df["label"]

# Train the logistic regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Test the model on the testing set
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)

# Predict the probability that a new text is AI-generated
new_text = "This is an example of AI-generated text."
X_new = vectorizer.transform([new_text])
prob = clf.predict_proba(X_new)[:, 1]
print("Probability of AI-generated text:", prob)
"""