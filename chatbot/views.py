import json, random, pickle, os
from django.http import JsonResponse
from django.shortcuts import render

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "intents.json")) as f:
    intents = json.load(f)

with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    clf = pickle.load(f)
with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

def home(request):
    return render(request, "chatbot/index.html")

def get_response(request):
    msg = request.GET.get("msg")
    X_test_vec = vectorizer.transform([msg])
    prediction = clf.predict(X_test_vec)[0]

    response = "Sorry, I didn’t get that 🤔"
    for intent in intents["intents"]:
        if intent["tag"] == prediction:
            response = random.choice(intent["responses"])
            break

    return JsonResponse({"response": response})
