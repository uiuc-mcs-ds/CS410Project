from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, ndcg_score
import numpy as np

def generate_features(bm25, queries, tokenized_tweets):
    features = []
    for query in queries:
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)  # BM25 scores for each tweet
        features.append(scores)
    return np.array(features).T  # Transpose to get tweets as rows

def generate_labels(weather_data, num_samples):
    description = weather_data["weather"][0]["description"].lower()
    humidity = weather_data["main"]["humidity"]
    wind_speed = weather_data["wind"]["speed"]

    # Determine labels
    labels = []
    if "rain" in description:
        labels.append("rain")
    if "snow" in description:
        labels.append("snow")
    if "cloud" in description:
        labels.append("cloudy")
    if "clear" in description:
        labels.append("clear")
    if humidity > 70:
        labels.append("very_humid")
    elif 50 <= humidity <= 70:
        labels.append("humid")
    else:
        labels.append("dry")
    if wind_speed > 25:
        labels.append("windy")
    elif wind_speed > 15:
        labels.append("breezy")

    # Default to "other" if no conditions match
    if not labels:
        labels.append("other")

    # Create the same label set for all samples
    return [labels] * num_samples

# Model Training
def train_model(X, y):
    # Convert multi-labels to a binary matrix
    mlb = MultiLabelBinarizer()
    y_binary = mlb.fit_transform(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    # Use MultiOutputClassifier with a base classifier (e.g., RandomForest)
    base_model = RandomForestClassifier()
    model = MultiOutputClassifier(base_model)
    model.fit(X_train, y_train)

    return model, X_test, y_test, mlb

# Model Evaluation
def evaluate_model(model, X_test, y_test, mlb):
    # Get predictions
    predictions = model.predict(X_test)

    # Decode binary predictions and true labels
    predictions_decoded = mlb.inverse_transform(predictions)
    y_test_decoded = mlb.inverse_transform(y_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)

    # For nDCG, convert decoded predictions back to binary matrix
    ndcg = ndcg_score(y_test, predictions, k=8)

    return accuracy, precision, recall, ndcg

