<!doctype html>
<html>
<body>
  <h3>MQTT Viewer</h3>
  <pre id="log"></pre>
  <script src="https://unpkg.com/mqtt/dist/mqtt.min.js"></script>
  <script>
    const log = (m)=>document.getElementById('log').textContent += m+"\n";
    const client = mqtt.connect("ws://YOUR_SERVER_IP:9001");  // ต้องเปิด listener 9001 แล้ว
    client.on("connect", ()=> {
      log("connected");
      client.subscribe("od/cam/cam_pi01");
      client.subscribe("status/cam/cam_pi01");
    });
    client.on("message", (topic, payload)=>{
      try { log(topic+" "+JSON.parse(new TextDecoder().decode(payload)).timestamp); }
      catch { log(topic+" "+new TextDecoder().decode(payload)); }
    });
  </script>
</body>
</html>
