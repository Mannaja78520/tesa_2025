## Download image to directory
import requests
import os

# URL ของภาพ
url = "https://aviationweek.com/sites/default/files/2022-11/drones-during-todays-pilot-2.jpg"

# สร้างโฟลเดอร์ images ถ้ายังไม่มี
os.makedirs("images", exist_ok=True)

# กำหนดชื่อไฟล์ปลายทาง
filename = "images/drones.jpg"

# ดาวน์โหลดและบันทึก
response = requests.get(url)
if response.status_code == 200:
    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"Downloaded to {filename}")
else:
    print(f"Failed to download (status code {response.status_code})")

