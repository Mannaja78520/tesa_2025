import json

data = {
            "robot_id": "TURTLEBOT3_01",
            "sensors": [
                {"name": "LIDAR", "status": "active", "range": 12.5},
                {"name": "IMU", "status": "active", "orientation": [0.0, 0.0, 0.1]},
                {"name": "WheelEncoder", "status": "inactive"}
            ],
            "battery": {
                "voltage": 12.3,
                "current": 0.5,
                "temperature": 36.7
            },
            "timestamp": "2025-11-09T22:30:00Z"
        }


with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)