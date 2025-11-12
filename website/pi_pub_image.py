import cv2, base64, json
ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
img_b64 = base64.b64encode(buf).decode("utf-8")
payload["thumb_b64"] = img_b64
client.publish(TOPIC_DATA, json.dumps(payload), qos=1)
