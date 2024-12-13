# Weather Data Text-Based Insights and Trend Analysis

## Overview
This project aims to combine weather data textual insights and trend analysis from weather reports and social media posts. By applying text mining techniques, trends will be extracted to build a model to predict weather conditions based on past events while incorporating text preprocessing techniques (tokenization, stop word removal, stemming), BM25 ranking, and feature extraction (TF-IDF) to analyze how textual descriptions of weather correlate with real weather data. nDCG@10 will be used to measure the effectiveness of ranked results. The data utilized will be weather dataset using OpenWeatherMap API and social media data from Twitter API. The coding will be completed using Python and various toolkits like Natural Language Toolkit for preprocessing text, Pandas for handling data, Scikit-learn for building a model and BM25 for ranking. The model performance is evaluated using accuracy, precision, recall, and nDCG@10 demonstrating trend prediction effectiveness through both metrics and visualizations.

Future enhancements may be done like addition of Flask application to build API for returning results, using more sophisticated models like Prophet for time-series analysis and enhancing the text data collection process using more robust APIs.

---

## Features

### Data Collection
- **Weather Data**:
  - Fetches real-time weather data using the [OpenWeather API](https://openweathermap.org/api).
  - Includes attributes such as temperature, humidity, wind speed, pressure and weather descriptions.
  - Supports location-based weather queries using latitude and longitude.
- **Social Media Data**:
  - Fetches recent tweets related to weather using the [Twitter API](https://developer.twitter.com/).
  - Filters tweets based on weather-related queries (e.g., "rain in New York").
  - Incorporates geographic filters for tweets if location information is provided.

### Data Preprocessing
- **Text Preprocessing**:
  - Tokenization: Splits textual data into tokens (words).
  - Stop word removal: Filters out commonly used words like "the" and "is."
  - Stemming: Reduces words to their root forms (e.g., "raining" to "rain").
- Normalizes numeric weather attributes for use in machine learning models.

### Feature Engineering
- Extracts term-frequency and inverse document frequency (TF-IDF) features for textual data.
- Combines numeric weather attributes into feature vectors for model training.
- Implements BM25 ranking for relevance scoring of text-based data.

### Predictive Modeling
- Trains a **Multi-Output Classifier** using `RandomForestClassifier` to predict multiple weather labels simultaneously.
- Implements train-test splitting and multi-label binarization for robust model evaluation.
- Utilizes metrics like:
  - **Accuracy**: Percentage of correctly predicted labels.
  - **Precision**: Proportion of true positives among predicted positives.
  - **Recall**: Proportion of true positives among actual positives.
  - **nDCG@10**: Measures ranking quality.

### Flexibility
- Combines data from APIs (OpenWeatherMap, Twitter) to demonstrate real-world integration.
- Extensible to more comprehensive datasets and APIs for advanced analysis.

### Visualization
- **Label Analysis**:
  - Label co-occurrence heatmap to visualize correlations between different weather labels.
  - Confusion matrices for individual labels to assess prediction accuracy.
- **Trend Analysis**:
  - Correlation heatmap to visualize relationships between weather attributes and textual features.
  - Bar charts summarizing model performance metrics.
- **Screenshots From Initial Testing**:

  <img width="500" alt="image" src="https://github.com/user-attachments/assets/18fc6612-8fa5-4cdf-994d-40948b222157" />
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/85413025-a853-4692-946d-eae129495c17" />
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/4de117f8-f817-44d9-9003-bea8249ff5ac" />
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/b237dd60-b4a0-4f34-8d4b-8b4e02371e95" />
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/8138ed55-065b-4afe-8a87-8e94eb3f630f" />
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/10821e1d-e07d-408d-9129-c0757b329c5a" />
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/1ee6c99f-16da-40b4-a548-40d5c22271e6" />
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/dc9747a5-da2b-4787-9fdd-8577e8bf82be" />
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/2b3cc12c-1a7c-4167-8169-49238018474f" />
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/f8616b0c-bd61-416b-b46b-0fb6eaa9ea2a" />
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/9c8d8c21-6b1e-49e3-867f-d94fef090589" />
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/5869aec7-967b-4c6e-91fb-a7f1844ba123" />

    
---

## Technology Stack
- **Programming Language**: Python
- **Libraries and Toolkits**:
  - `NumPy` and `Pandas` for data manipulation and analysis.
  - `scikit-learn` for machine learning and performance evaluation.
  - `matplotlib` and `seaborn` for data visualization.
  - `NLTK` for text preprocessing.
  - `Tweepy` for accessing the Twitter API.

---

## Key Advantages
- Demonstrates the integration of structured (numeric weather data) and unstructured (textual descriptions) datasets.
- Provides a comprehensive pipeline from data collection and preprocessing to prediction and visualization.
- Modular design, allowing easy extension with advanced models and datasets.
- Supports multi-label classification for more realistic weather predictions.

---

## Getting Started
- Clone the repository:  
  ```bash
  git clone https://github.com/uiuc-mcs-ds/CS410Project.git

---

## Installation Instructions

Follow these instructions to set up the environment and run the Python project.

### Prerequisites
- Python 3.7.9 or later installed on your system.
- `pip` (Python package installer) should be installed and accessible.
- Basic knowledge of Python virtual environments (recommended).

### Steps to Install

1. **Clone the Repository**
   ```bash
   git clone https://github.com/uiuc-mcs-ds/CS410Project.git
   cd CS410Project

2. **Set Up a Virtual Environment (Recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # On macOS/Linux
   .\venv\Scripts\activate     # On Windows

3. **Install Required Dependencies Install all dependencies listed in the requirements.txt file**
   ```bash
   pip install -r requirements.txt

4. **Install NLTK Data Download necessary datasets for NLTK (e.g., stopwords and punkt)**
   ```bash
   python -m nltk.downloader punkt
   python -m nltk.downloader stopwords

5. **Set Up API Keys**
   Ensure you have the required API keys for:
    - OpenWeatherMap API for weather data.
    - Twitter API (Tweepy) for tweet collection.
   
   Place your API keys in the appropriate variables in the script files where they are required.

6. **Run the Python Code Execute the main script or specific modules as needed**
   ```bash
   python data_collection.py

7. **Verify Outputs**
    - Check the console for logs and outputs.
    - Ensure that the processed data, model results, and visualizations are generated as expected.

