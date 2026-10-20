import json
import logging
import os

import paho.mqtt.client as mqtt

BROKER = os.getenv("MQTT_BROKER", "localhost")
PORT = int(os.getenv("MQTT_PORT", "1883"))
TOPIC = os.getenv("MQTT_TOPIC", "pump/01/state")


def on_connect(client, userdata, flags, reason_code, properties):
    logging.info("[MQTT] Connected to %s:%s", BROKER, PORT)
    client.subscribe(TOPIC)
    logging.info("[MQTT] Subscribed to %s", TOPIC)


def on_message(client, userdata, message):
    payload = message.payload.decode("utf-8")
    try:
        parsed = json.loads(payload)
        print("[MQTT] Received:", parsed)
    except Exception:
        print("[MQTT] Raw message:", payload)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, 60)
    client.loop_forever()
