# pub_image.py
import cv2, paho.mqtt.client as mqtt

BROKER="192.168.50.242"; CAM_ID="cam_01"
TOPIC=f"od/cam/{CAM_ID}/image"

cli=mqtt.Client(mqtt.CallbackAPIVersion.VERSION2); cli.connect(BROKER,1883,30)
frame=cv2.imread("img_0004.jpg")
ok,buf=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,75])
cli.publish(TOPIC, buf.tobytes(), qos=1)
cli.disconnect()
