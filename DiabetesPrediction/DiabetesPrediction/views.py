
from django.shortcuts import render
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
def home(request):
    return render(request, "home.html")

def predict(request):
    return render(request, "predict.html")

def result(request):
    """Predicts diabetes based on user input and loaded model.

    This function expects eight numeric values (n1 to n8) from the user's GET request
    and uses a pre-trained model loaded from 'diabete_prediction_model.pkl' to
    predict diabetes risk. It returns a dictionary containing the prediction
    ("Positif" or "Negatif") as "result2" to be rendered in the "predict.html" template.
    """

    # Load the pre-trained model
    with open('../../diabete_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Extract user input from the GET request
    try:
        val1 = float(request.GET['n1'])
        val2 = float(request.GET['n2'])
        val3 = float(request.GET['n3'])
        val4 = float(request.GET['n4'])
        val5 = float(request.GET['n5'])
        val6 = float(request.GET['n6'])
        val7 = float(request.GET['n7'])
        val8 = float(request.GET['n8'])
    except (KeyError, ValueError):
        # Handle potential errors (missing keys or non-numeric values)
        return render(request, "predict.html", {"result2": "Erreur: Données insuffisantes pour predire votre etat"})

    # Prepare the input data for prediction
    data = [[val1, val2, val3, val4, val5, val6, val7, val8]]

    # Make prediction using the loaded model
    pred = model.predict(data)

    # Translate prediction to user-friendly outcome
    result1 = "Positif" if pred[0] == 1 else "Negatif"

    # Return the prediction for rendering in the template
    return render(request, "predict.html", {"result2": result1})