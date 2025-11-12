import json, paho.mqtt.client as mqtt
cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2); cli.connect("192.168.50.242",1883) # <-- ใส่ IP คอมที่รัน mosquitto
cmd = {"cmd":"set_rate","hz":5}
cli.publish("cmd/cam/cam_01", json.dumps(cmd), qos=1)
