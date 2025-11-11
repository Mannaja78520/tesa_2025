import requests

url = "https://tesa-api.crma.dev/api/object-detection/bb120e02-dc26-48ae-a876-f17e7fb2373a"

headers = {
    "x-camera-token": "94bb9161d9bb03fb0d4382ec7a2413379e183c386ccbc89934f83d8cb6c2a651"   # ใส่ token จริงของกล้อง
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    print("✅ Success!")
    print(data)
elif response.status_code == 401:
    print("❌ Unauthorized — invalid camera token")
else:
    print(f"Error {response.status_code}: {response.text}")
