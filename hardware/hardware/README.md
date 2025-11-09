Hardware Setup Guide
====================

This branch documents everything needed to bring the Raspberry Pi–based hardware
platform online at full performance. The checklist below mirrors the raw notes
in `setup.txt`, but groups them by intent so the pull request stays easy to
review.

1. Update the base OS and firmware
----------------------------------
```bash
sudo apt update && sudo apt full-upgrade -y
sudo rpi-eeprom-update -a
```

2. Enable zero-conf networking
------------------------------
```bash
sudo apt install -y avahi-daemon avahi-utils
sudo systemctl enable --now avahi-daemon
```

3. Install Python and ML dependencies
-------------------------------------
```bash
sudo apt-get install -y python3-pip libjpeg-dev libopenblas-dev \
    libopenmpi-dev libomp-dev
pip install pandas scikit-learn matplotlib openpyxl
pip install opencv-python ultralytics torch onnxruntime
pip install torch torchvision torchaudio
pip install pandas matplotlib seaborn scikit-learn tensorflow keras \
    flask django opencv-python beautifulsoup4 requests
```

4. Force the CPU into performance mode
--------------------------------------
```bash
sudo apt install -y cpufrequtils
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
sudo systemctl disable ondemand
sudo systemctl enable --now cpufrequtils
```

5. Reduce power draw (optional)
-------------------------------
Add the following snippets inside `/boot/firmware/config.txt` to disable HDMI
and LEDs, and to configure the cooling fan:
```
avoid_warnings=2
arm_boost=1
dtparam=cooling_fan=on
dtparam=fan_temp0=5000
dtparam=fan_temp0_speed=255
```

6. Disable unattended upgrades if you need deterministic reboots
----------------------------------------------------------------
Set every toggle to `0` inside `/etc/apt/apt.conf.d/20auto-upgrades` and mask
`systemd-networkd-wait-online.service`:
```bash
sudo systemctl disable systemd-networkd-wait-online.service
sudo systemctl mask systemd-networkd-wait-online.service
```

7. Optional developer tools
---------------------------
```bash
sudo apt install -y npm nodejs docker.io
```

8. TensorFlow Lite C++ toolchain (when building native demos)
-------------------------------------------------------------
Follow the sequence below whenever you need the full TFLite C++ stack:
```bash
sudo apt install -y curl unzip zip clang git cmake build-essential \
    python3-dev python3-pip protobuf-compiler libprotoc-dev libgtest-dev

curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.18.0/bazelisk-linux-arm64
chmod +x bazelisk-linux-arm64
sudo mv bazelisk-linux-arm64 /usr/local/bin/bazel

git clone --depth 1 https://github.com/tensorflow/tensorflow.git
cd tensorflow
./configure
bazel clean --expunge
bazel build -c opt --config=clang_local //tensorflow/lite:libtensorflowlite.so --verbose_failures
sudo cp bazel-bin/tensorflow/lite/libtensorflowlite*.so /usr/local/lib/
sudo mkdir -p /usr/local/include/tensorflow/
sudo cp -r tensorflow/* /usr/local/include/tensorflow/
```

For FlatBuffers:
```bash
git clone https://github.com/google/flatbuffers.git
cd flatbuffers
git checkout v24.3.25
cmake -G "Unix Makefiles"
make -j$(nproc)
sudo make install
```

Keep this file in sync with `setup.txt` whenever the hardware workflow changes.
