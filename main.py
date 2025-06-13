# import require libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score , classification_report

# we create a small dataset of email text ans labels (0 for not spam, 1 for spam)
emails = {
    "Get rich Quick! Click here to win a million dollars!",
    "Hello, could you please review this document for me",
    "Discounts on luxuary watches and handbages!",
    "Meeting scheduled for tommorow , please confirm your attendance.",
    "Congratulations, you've won a free gift card!",
}

labels = [1,0,1,0,1]

# convert text data into numerical features using count vectorization
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(emails)

# split the data into training and testing sets
X_train, x_test, y_train, y_test = train_test_split(x, labels,test_size=0.2)

# create a multionomial naive bayes classifier
model = MultinomialNB()

# train the model on training data
model.fit(X_train, y_train)

# make prediction on test data
y_pred = model.predict(x_test)

# evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy: ", accuracy)
print("Classification Report:\n", report)

#  predict whether a new email is spam or not
new_email = ["You've won a free cruise vacation"]
new_email_vectorized = vectorizer.transform(new_email)
predicted_label = model.predict(new_email_vectorized)

if predicted_label[0] == 0:
    print("Predicted as not spam.")
else:
    print("Predicted as spam.")
