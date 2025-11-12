import json, paho.mqtt.client as mqtt
def on_msg(c,u,m): print(m.topic, json.loads(m.payload))
cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
cli.on_message = on_msg
cli.connect("192.168.50.242",1883) # <-- ใส่ IP คอมที่รัน mosquitto
cli.subscribe("od/cam/#", qos=1)
cli.loop_forever()