# 🚨 Emergency Vehicle Detection App using YOLOv8 + CLIP

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Built%20with-Gradio-orange.svg)](https://gradio.app/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Real--Time-green.svg)](https://github.com/ultralytics/ultralytics)
[![CLIP](https://img.shields.io/badge/CLIP-OpenAI-lightgrey.svg)](https://github.com/openai/CLIP)

A real-time web application for detecting **emergency vehicles** (ambulance, fire truck, police car) using **YOLOv8** for object detection and **CLIP** for zero-shot image-text matching. Built with **Gradio**, this app alerts the user with sound notifications and displays their **live location on Google Maps**.

---

## 📸 Demo

> Real-time detection using webcam feed with audio + visual alerts. Bounding boxes only appear for confirmed emergency vehicles.
> <br>
> <b>NOTE:</b> Click the image below for demo.
<a href="https://jatinnama7-ambulance-detction-system.hf.space/" target="_blank">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExd25icnIybXgycjh4dmJrN2cxdHphZ2dpM2g1cWN2bnhnYXB5YjdsaCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/L1R1tvI9svkIWwpVYr/giphy.gif" 
       alt="Demo GIF" 
       style="width:400px; height:auto; border:none;" />
</a>

---

## 🧠 Features

- 🔍 **YOLOv8 + CLIP**: Combines object detection with semantic validation to detect emergency vehicles only.
- 📸 **Real-time Webcam Processing**: Processes frames smoothly from your webcam.
- 🎯 **Accurate Filtering**: Uses CLIP to ensure high precision, filtering only emergency-related objects.
- 🔊 **Sound Alerts**: Plays a warning sound whenever an emergency vehicle is detected.
- 🌐 **Google Maps Integration**: Shows your live location for context-aware alerts.
- 📦 **Gradio UI**: Easy-to-use browser interface to start/stop detection.

---

## 🧰 Tech Stack

| Component        | Library/Tool           |
| ---------------- | ---------------------- |
| Object Detection | YOLOv8 (`ultralytics`) |
| Text-Image Match | CLIP (OpenAI)          |
| GUI              | Gradio                 |
| Alerts           | Pygame Sound / Beep    |
| Webcam Feed      | OpenCV                 |
| Mapping          | Google Maps (iframe)   |

---

## 🖥️ Installation

### 1️⃣ Clone the Repo

```bash
git clone https://github.com/jatinnama7/AI-Ambulance-Detection
cd AI-Ambulance-Detection
```

### 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

> 💡 Make sure you're using **Python 3.8 or later** and have a virtual environment activated (optional but recommended).

### 3️⃣ Run the App

```bash
python app.py
```

### 4️⃣ Open in Browser

> Gradio will provide a `http://127.0.0.1:7860` link. Click it or open in your browser.

---

## 📂 Folder Structure

```
AI-Ambulance-Detection
├── app.py                # Main Gradio application
├── requirements.txt      # Required Python libraries
└── README.md            
```

---

## 🔔 Sound Alert Instructions

- If you have a custom **ambulance_alert.wav** or **.mp3** file, place it in the project root.
- If missing, the app plays a default beep as fallback.
- Sound plays only once per detection to avoid spamming.

---

## 🌍 Location Support (Google Maps)

- Your **public IP** is used to retrieve approximate location.
- A Google Maps iframe is embedded below the detection interface.
- Internet connection is required for maps to work.

---

## ⚙️ Configuration

<details>
<summary><strong>Click to expand</strong> 🔧</summary>

- **Detection Confidence Threshold**: Tuned in YOLOv8 model (default 0.3).
- **CLIP Matching Text**: Hardcoded emergency terms like `"ambulance"`, `"fire truck"`, `"police car"`.
- **Alert Interval**: Alerts only once per vehicle to avoid repeat spam.

</details>

---

## 📦 Requirements.txt

```txt
gradio
opencv-python
torch
numpy
Pillow
ultralytics
pygame
ftfy
regex
tqdm
git+https://github.com/openai/CLIP.git
```

---

## 🧪 Tested On

| System      | GPU                                 | Works? |
| ----------- | ----------------------------------- | ------ |
| Windows 10  | NVIDIA GTX 1650                     | ✅     |
| Mac (M1/M2) | ❌ (No official YOLOv8 GPU support) |

---

## 🚧 Limitations

- Requires decent webcam + lighting for reliable results.
- Relies on images shown on-screen or real-world visuals.
- Approximate geolocation only (via IP, not GPS).
- Not trained for regional emergency vehicle variations.

---

## ✨ Future Enhancements

- 📱 Mobile support via Gradio share links.
- 📈 Detection analytics dashboard.
- 🧠 Integration with custom-trained emergency vehicle classifier.
- 📍  Real-time GPS-based tracking (via browser/device API).

---

## 🤝 Contributing

Pull requests are welcome! If you have suggestions, open an issue or PR.

---

## 📝 License

MIT License. Use freely for research, education, or personal use.

---

## 💬 Contact

Made with ❤️ by Jatin Nama.  
📧 Email: jatinnama7@gmail.com  
🌐 [GitHub](https://github.com/jatinnama7)

