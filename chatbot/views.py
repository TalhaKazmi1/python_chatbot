import json, random, pickle, os
from django.http import JsonResponse
from django.shortcuts import render
from rapidfuzz import process

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load intents
with open(os.path.join(BASE_DIR, "intents.json"), encoding="utf-8") as f:
    intents = json.load(f)

# Load trained model + vectorizer
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    clf = pickle.load(f)
with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)


def home(request):
    return render(request, "chatbot/index.html")


def generate_response(user_message, model, vectorizer, data):
    """
    Handle fuzzy matching + ML + fallback logic with confidence
    """
    # 1. Fuzzy match
    best_match, score, _ = process.extractOne(
        user_message,
        [p for intent in data["intents"] for p in intent["patterns"]]
    )

    if score > 80:  # Strong fuzzy match
        for intent in data["intents"]:
            if best_match in intent["patterns"]:
                return random.choice(intent["responses"])

    # 2. ML prediction with confidence check
    X_test_vec = vectorizer.transform([user_message])
    proba = model.predict_proba(X_test_vec)[0]   # probability distribution
    max_prob = max(proba)
    prediction = model.classes_[proba.argmax()]

    if max_prob >= 0.6:  # <-- confidence threshold
        for intent in data["intents"]:
            if intent["tag"] == prediction:
                return random.choice(intent["responses"])

    # 3. Fallback
    return "🤔 I don’t know about that, but I can tell you a lot about the 2022 World Cup!"


def get_response(request):
    """
    Django view: takes ?msg=, returns JSON response
    """
    msg = request.GET.get("msg", "")
    if not msg.strip():
        return JsonResponse({"response": "Please type something!"})

    response = generate_response(msg, clf, vectorizer, intents)
    return JsonResponse({"response": response})
