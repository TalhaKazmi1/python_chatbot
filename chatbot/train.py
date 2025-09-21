import json, pickle, os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "intents.json"), encoding="utf-8") as f:
    data = json.load(f)

X_train, y_train = [], []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        X_train.append(pattern)
        y_train.append(intent["tag"])

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

clf = LogisticRegression()
clf.fit(X_train_vec, y_train)

with open(os.path.join(BASE_DIR, "model.pkl"), "wb") as f:
    pickle.dump(clf, f)
with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained and saved!")
