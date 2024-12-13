import requests
import random
import tweepy
import json

# Fetch Weather Data
# Requires api key that generated from OpenWeatherAPI account page
def fetch_weather_data(latitude, longitude, api_key):
    url = f'http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=imperial'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Unable to fetch weather data"}

# Fetch Twitter Data
# Requires api key, secret that generated from Twitter account page
def fetch_tweets(api_key, api_secret, bearer_token, query, latitude=None, longitude=None, radius="25mi"):
    client = tweepy.Client(bearer_token=bearer_token)
    #if latitude and longitude:
    #    query += f" point_radius:[{longitude} {latitude} {radius}]"
    #response = client.search_recent_tweets(query=query, max_results=10, tweet_fields=["text", "created_at"])
    query = "weather New York"
    response = client.search_recent_tweets(query=query, max_results=50, tweet_fields=["text", "created_at"])
    return [tweet.text for tweet in response.data] if response.data else []

def cache_data(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_cached_data(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# Generate diverse simulated weather data
def generate_simulated_weather_data(num_samples=50):
    weather_descriptions = [
        "light rain", "clear sky", "snow", "few clouds", "overcast clouds",
        "thunderstorm", "drizzle", "mist", "fog", "heavy snow"
    ]
    simulated_data = []
    for _ in range(num_samples):
        description = random.choice(weather_descriptions)
        humidity = random.randint(30, 100)  # Random humidity between 30% and 100%
        wind_speed = random.randint(5, 40)  # Random wind speed between 5 km/h and 40 km/h
        temperature = random.randint(-10, 40)  # Random temperature between -10°C and 40°C
        pressure = random.randint(950, 1050)  # Random pressure between 950 hPa and 1050 hPa
        simulated_data.append({
            "weather": [{"description": description}],
            "main": {"humidity": humidity, "temp": temperature, "pressure": pressure},
            "wind": {"speed": wind_speed},
        })
    return simulated_data

# Generate feature vectors from simulated data
def generate_dummy_features(simulated_data):
    feature_vectors = []
    for data in simulated_data:
        description = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        temperature = data["main"]["temp"]
        pressure = data["main"]["pressure"]

        # Encode description into numerical categories (one-hot encoding or simple integers)
        description_mapping = {
            "light rain": 1, "clear sky": 2, "snow": 3, "few clouds": 4,
            "overcast clouds": 5, "thunderstorm": 6, "drizzle": 7, "mist": 8,
            "fog": 9, "heavy snow": 10
        }
        desc_encoded = description_mapping.get(description, 0)  # Default to 0 if not found

        # Create a feature vector [desc_encoded, humidity, wind_speed, temperature]
        feature_vector = [humidity, wind_speed, temperature, pressure]
        feature_vectors.append(feature_vector)
    return feature_vectors
