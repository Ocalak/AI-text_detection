import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# read in the dataset
data = pd.read_csv("dataset.csv")

# create a CountVectorizer to convert the text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["text"])
y = data["label"]

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create a Naive Bayes classifier
clf = MultinomialNB()

# train the classifier on the training data
clf.fit(X_train, y_train)

# make predictions on the test data
y_pred = clf.predict(X_test)

# calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)




#https://mr-collins-llb.medium.com/implementing-ai-text-generation-detection-with-python-a-step-by-step-guide-using-machine-learning-83e425c6a737