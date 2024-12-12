# Weather Data Text-Based Insights and Trend Analysis

## Overview
This project aims to combine weather data textual insights and trend analysis from weather reports and social media posts. By applying text mining techniques, trends will be extracted to build a model to predict weather conditions based on past events while incorporating text preprocessing techniques (tokenization, stop word removal, stemming), BM25 ranking, and feature extraction (TF-IDF) to analyze how textual descriptions of weather correlate with real weather data. nDCG@10 will be used to measure the effectiveness of ranked results. The data utilized will be weather dataset using OpenWeatherMap API and social media data from Twitter API. The coding will be completed using Python and various toolkits like Natural Language Toolkit for preprocessing text, Pandas for handling data, Scikit-learn for building a model and BM25 for ranking. The model performance is evaluated using accuracy, precision, recall, and nDCG@10 demonstrating trend prediction effectiveness through both metrics and visualizations.

Future enhancements may be done like addition of Flask application to build API for returning results, using more sophisticated models like Prophet for time-series analysis and enhancing the text data collection process using more robust APIs.

---

## Features

### Data Collection
- **Weather Data**:
  - Fetches real-time weather data using the [OpenWeather API](https://openweathermap.org/api).
  - Includes attributes such as temperature, humidity, wind speed, and weather descriptions.
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

### Visualization
- **Label Analysis**:
  - Label co-occurrence heatmap to visualize correlations between different weather labels.
  - Confusion matrices for individual labels to assess prediction accuracy.
- **Trend Analysis**:
  - Correlation heatmap to visualize relationships between weather attributes and textual features.
  - Bar charts summarizing model performance metrics.

### Flexibility
- Combines data from APIs (OpenWeatherMap, Twitter) to demonstrate real-world integration.
- Extensible to more comprehensive datasets and APIs for advanced analysis.

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


