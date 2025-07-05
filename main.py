
from src.model import load_data, train_model, predict_sales

# Load and train
data = load_data()
model, score = train_model(data)

print(f"Model Accuracy: {score:.2f}")

# Example Prediction
example_input = [230.1, 37.8, 69.2]
predicted = predict_sales(model, example_input)
print(f"Predicted Sales: {predicted:.2f}")
