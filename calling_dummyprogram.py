import requests

url = "http://127.0.0.1:5000/predict"
data = {"news": "The stock market crashed today due to economic instability."}

response = requests.post(url, json=data)
print(response.json())  # Output: {"news": "...", "sentiment": "negative"}
