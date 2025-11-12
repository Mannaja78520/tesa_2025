import json, paho.mqtt.client as mqtt
cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2); cli.connect("YOUR_SERVER_IP",1883)
cmd = {"cmd":"set_rate","hz":5}
cli.publish("cmd/cam/cam_01", json.dumps(cmd), qos=1)
