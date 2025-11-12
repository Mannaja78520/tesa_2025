import paho.mqtt.client as mqtt

BROKER = "192.168.50.242"
CAM_ID = "cam_01"
TOPIC_IMG = f"od/cam/{CAM_ID}/image"

def on_msg(c,u,m):
    print("received bytes:", len(m.payload))
    with open("recv.jpg", "wb") as f:
        f.write(m.payload)
    print("saved -> recv.jpg")

cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="subimg")
cli.on_message = on_msg
cli.connect(BROKER, 1883, 30)
cli.subscribe(TOPIC_IMG, qos=1)
cli.loop_forever()
