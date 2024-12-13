import pandas as pd
from data_collection import cache_data, fetch_weather_data, fetch_tweets, load_cached_data, generate_simulated_weather_data, generate_dummy_features
from preprocessing import preprocess_text, tfidf_vectorize, bm25_ranking
from model import generate_features, generate_labels, train_model, evaluate_model
from visualization import plot_label_confusion_matrix, plot_label_cooccurrence, plot_model_performance, visualize_trends

# Main function
def main():
    
    ########data_collection.py
    # Example Usage
    # weather_data = fetch_weather_data('New York', 'your_openweathermap_api_key')
    cache_file_weather = "weather_data.json"
    cache_file_tweets = "tweets.json"
    # Check if cached data exists and is not too old
    cache_age_limit = 360000  # 1 hour in seconds
    weather_data = load_cached_data(cache_file_weather)
    #if weather_data is None or time.time() - weather_data['timestamp'] > cache_age_limit:
    #    weather_data = fetch_weather_data(40.7128,-74.0060,'your_openweathermap_api_key')
    #    cache_data(cache_file_weather, {'timestamp': time.time(), 'data': weather_data})
    tweets = load_cached_data(cache_file_tweets)
    #if tweets is None or time.time() - tweets['timestamp'] > cache_age_limit:
    #    tweets = fetch_tweets(
    #        api_key='your_twitter_api_key',
    #        api_secret='your_twitter_api_secret',
    #        bearer_token='your_twitter_api_bearer_token',
    #        query='weather New York'
    #        #latitude=40.7128,  # Latitude for New York
    #        #longitude=-74.0060,  # Longitude for New York
    #        #radius="25mi"  # Search radius in miles
    #    )
    #    cache_data(cache_file_tweets, {'timestamp': time.time(), 'data': tweets})
    weather_data=weather_data["data"]
    tweets=tweets["data"]
    # Use the cached weather_data and tweets
    print("***weather_data***",weather_data)
    print("***tweets***",tweets)
    #weather_data = fetch_weather_data(40.7128,-74.0060,'your_openweathermap_api_key')
    #print(weather_data)
    #tweets = fetch_tweets(
    #    api_key='your_twitter_api_key',
    #    api_secret='your_twitter_api_secret',
    #    bearer_token='your_twitter_api_bearer_token',
    #    query='weather New York'
        #latitude=40.7128,  # Latitude for New York
        #longitude=-74.0060,  # Longitude for New York
        #radius="25mi"  # Search radius in miles
    #)
    #print(tweets)

    ########preprocessing.py
    cleaned_tweets = [preprocess_text(tweet) for tweet in tweets]
    print("***cleaned_tweets***",cleaned_tweets)
    tfidf_vectors = tfidf_vectorize(cleaned_tweets)
    print("***tfidf_vectors***",tfidf_vectors)
    bm25 = bm25_ranking(cleaned_tweets)
    print("***bm25***",bm25)

    ########model.py
    queries = ["rain", "snow", "wind", "clouds", "clear"]
    tokenized_tweets = [tweet.split() for tweet in cleaned_tweets]
    X = generate_features(bm25, queries, tokenized_tweets)
    #print("***X***",X)
    y = generate_labels(weather_data, len(X))
    simulated_weather_data = generate_simulated_weather_data(56432)
    y = [generate_labels(weather_data, 1)[0] for weather_data in simulated_weather_data]
    X = generate_dummy_features(simulated_weather_data)
    # print("***Weather Data***", simulated_weather_data)
    # print("***X (Feature Vectors)***", X)
    # print("***y (Labels)***", y)
    model, X_test, y_test, mlb = train_model(X, y)
    # print("***model***",model)
    # print("***X_test***",X_test)
    # print("***y_test***",y_test)
    # print("***model***",model)
    metrics = evaluate_model(model, X_test, y_test, mlb)
    # print("***metrics***",metrics)

    ########visualization.py
    y_test_binary = mlb.transform(y_test)
    # print("***y_test_binary***")
    y_pred_binary = model.predict(X_test)
    # print("***y_pred_binary***")
    plot_label_confusion_matrix(y_test_binary, y_pred_binary, mlb.classes_)
    plot_label_cooccurrence(mlb.transform(y), mlb.classes_)
    plot_model_performance(metrics, ["Accuracy", "Precision", "Recall", "nDCG@10"])
    feature_data = pd.DataFrame(
        X,
        columns=["Humidity", "Wind_Speed", "Temperature", "Pressure"]
    )
    visualize_trends(feature_data)

# Entry point
if __name__ == "__main__":
    main()
