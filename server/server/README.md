Server & Streaming Notes
========================

Use this branch to keep all server-side and streaming tweaks scoped to a single
pull request. The snippets below were lifted from `setup.txt` and restructured
so reviewers get a clean checklist.

Camera streaming
----------------
```bash
sudo apt update
sudo apt install -y mjpg-streamer
```
- Point MJPG-streamer to `/dev/video0` for a quick web preview while tuning.
- For ad‑hoc tests, install `cheese` and launch it directly on the device.

Remote visualization from a laptop
----------------------------------
On the laptop (client):
```bash
sudo apt install -y xauth xorg
ssh -X user@<ip_server>
# or compress the stream if bandwidth is tight
ssh -YC user@<ip_server>
```

On the server (camera side):
```bash
sudo apt install -y cheese
cheese
```

Cron reliability checklist
--------------------------
```bash
sudo systemctl status cron
sudo systemctl enable cron
sudo systemctl start cron
crontab -e   # add entries such as: @reboot <command>
```

General Python stack
--------------------
Keep these runtime dependencies handy for server-side scripts:
```bash
pip3 install setuptools numpy Cython
pip3 install torch torchvision
pip install pandas matplotlib seaborn scikit-learn tensorflow keras \
    flask django opencv-python beautifulsoup4 requests
```

Feel free to extend this document with deployment scripts or systemd units as
they become part of the server PR.
