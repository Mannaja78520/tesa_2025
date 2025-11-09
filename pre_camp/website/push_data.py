import json
import requests

# โหลดไฟล์ JSON จากเครื่อง
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ส่งข้อมูลไปยังเว็บ local
url = "http://localhost:5000/upload"
response = requests.post(url, json=data)

print("สถานะ:", response.status_code)
print("ผลลัพธ์จากเซิร์ฟเวอร์:", response.text)
