// HoopStream.jsx
import React from "react";

export default function HoopStream() {
  // ถ้า React รันบนเครื่องอื่น ให้เปลี่ยนเป็น IP ของ server Flask
  const STREAM_URL = process.env.REACT_APP_STREAM_URL || "http://192.168.50.50:5000/video";

  return (
    <div
      style={{
        backgroundColor: "#020617",
        minHeight: "100vh",
        margin: 0,
        padding: "1rem",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        color: "#e5e7eb",
        boxSizing: "border-box",
      }}
    >
      <h2 style={{ marginBottom: "1rem" }}>Hoop Detection Stream</h2>
      <img
        src={STREAM_URL}
        alt="Hoop detection stream"
        style={{
          maxWidth: "90vw",
          maxHeight: "80vh",
          border: "2px solid #22c55e",
          borderRadius: "12px",
          display: "block",
        }}
      />
    </div>
  );
}
