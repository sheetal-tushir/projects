import os
import gradio as gr
import joblib

# Load model
deployed_dt = joblib.load("diabetes_prediction_model.pkl")

def predict_diabetes(pregnancies, glucose, insulin, bmi, age):
    input_data = [[pregnancies, glucose, insulin, bmi, age]]
    prediction = deployed_dt.predict(input_data)

    if prediction[0] == 1:
        return "Prediction: High Risk of Diabetes (Positive)"
    else:
        return "Prediction: Low Risk of Diabetes (Negative)"

interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Age"),
    ],
    outputs=gr.Text(label="Assessment Result"),
    title="Diabetes Prediction System",
    description="Enter the medical metrics to predict diabetes risk."
Developed by: Sheetal

Panipat Institute of Engineering & Technology (PIET)

Enter the medical metrics to predict diabetes risk using a Decision Tree Machine Learning model.
"""
)

if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
