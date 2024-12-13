import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

#Plot label confusion matrix
def plot_label_confusion_matrix(y_test, y_pred, labels):
    confusion_matrices = multilabel_confusion_matrix(y_test, y_pred)
    for i, label in enumerate(labels):
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrices[i], display_labels=[f"Not {label}", label])
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix for '{label}'")
        plt.show()

#Plot label cooccurence
def plot_label_cooccurrence(y, labels):
    cooccurrence = np.dot(np.array(y).T, np.array(y))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cooccurrence, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title("Label Co-occurrence Heatmap")
    plt.xlabel("Labels")
    plt.ylabel("Labels")
    plt.show()

def plot_model_performance(metrics, labels):
    accuracy, precision, recall, ndcg = metrics
    values = [accuracy, precision, recall, ndcg]
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color='skyblue')
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model Performance Metrics")
    plt.show()

def visualize_trends(df):
    sns.heatmap(df.corr(), annot=True)
    plt.title('Weather Data and Text Features Correlation')
    plt.show()
