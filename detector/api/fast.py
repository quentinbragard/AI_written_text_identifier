import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from detector import main

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(
        text_input: str = main.get_demo_text(api=True)
        ):
    """
    Predict if the input text is AI written or not. Returns a dictionary with the prediction and the probability.
    """

    proba, class_pred = main.get_prediction(text_input=text_input, api=True)

    return {"Probability": float(proba),
            "Prediction": class_pred}


@app.get("/")
def root():
    return {"greeting": "Hello"}
