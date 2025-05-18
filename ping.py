import requests

url = "http://192.168.10.179:5000/tracks"
params = {
    "youtube": "https://www.youtube.com/watch?v=d6GjSxPMq0w",
    "url": "https://open.spotify.com/playlist/7Adu8sIhfrotX2cbQkT3bP?si=0332efd206234b30"
}

try:
    response = requests.get(url, params=params, timeout=10)
    print("✅ Status Code:", response.status_code)
    print("✅ Response:", response)
except requests.exceptions.RequestException as e:
    print("❌ Error:", e)