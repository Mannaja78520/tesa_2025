#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup

# ดึงหน้าเว็บ
url = "https://raw.githubusercontent.com/Mannaja78520/tesa_2025/main/README.md"
response = requests.get(url)
if response.status_code == 200:
    print(response.text)
else:
    print("โหลดไม่สำเร็จ:", response.status_code)

# url = "http://localhost:8000"
# response = requests.get(url)
# soup = BeautifulSoup(response.text, 'html.parser')
# print(soup.title.text)


# หา headline ทั้งหมด
# headlines = soup.find_all('a', class_='DY5T1d')
# for h in headlines[:5]:
#     print(h.text)

# โปรแกรมนี้จะ “เข้าเว็บข่าว Google News” แล้วดึงชื่อข่าว 5 อันดับแรกออกมา
# โดยใช้ 2 ไลบรารี:

# requests → โหลดเว็บ

# BeautifulSoup → อ่าน HTML และค้นแท็ก