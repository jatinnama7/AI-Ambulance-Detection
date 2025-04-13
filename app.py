import gradio as gr
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import clip
import threading
import pygame
import time
import os

# Load models
print("Loading YOLOv8 model...")
yolo_model = YOLO("yolov8n.pt")
print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Define emergency vehicle categories
categories = ["an emergency vehicle", "a fire truck", "a police car", "an ambulance", "a police bike"]
text_prompt = clip.tokenize(categories).to(device)

# Global variables
camera_active = True
emergency_count = 0
camera_index = 0  # Default camera (usually built-in webcam)

# Try to load audio file, create a fallback if not found
try:
    pygame.mixer.init()
    audio_enabled = True
    print("Audio enabled")
    
    # Create a simple alert sound (properly formatted for stereo)
    alert_sound = None
    if os.path.exists("ambulance_alert.mp3"):
        try:
            alert_sound = pygame.mixer.Sound("ambulance_alert.mp3")
            print("Loaded MP3 alert sound")
        except Exception as e:
            print(f"Error loading MP3: {e}")
    elif os.path.exists("ambulance_alert.wav"):
        try:
            alert_sound = pygame.mixer.Sound("ambulance_alert.wav")
            print("Loaded WAV alert sound")
        except Exception as e:
            print(f"Error loading WAV: {e}")
    
    # If no sound file is found or loaded, create a fallback sound
    if alert_sound is None:
        print("Creating fallback alert sound")
        sample_rate = 22050
        duration = 1.0  # 1 second
        frequency = 440  # A4 note
        
        # Create stereo sound (2D array)
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * frequency * t) * 32767
        stereo_tone = np.column_stack((tone, tone)).astype(np.int16)
        
        alert_sound = pygame.mixer.Sound(pygame.sndarray.make_sound(stereo_tone))
        print("Created fallback stereo alert sound")
    
except Exception as e:
    audio_enabled = False
    alert_sound = None
    print(f"Warning: Could not initialize audio: {e}. Audio alerts disabled.")

def play_alert():
    """Play audio alert when emergency vehicle detected"""
    if audio_enabled and alert_sound:
        try:
            # Use the Sound object directly
            if not pygame.mixer.get_busy():
                alert_sound.play()
                print("Playing alert sound")
        except Exception as e:
            print(f"Error playing alert: {e}")
            try:
                # Last resort system beep
                print("\a")
            except:
                pass

# Function to toggle camera
def toggle_camera(active):
    global camera_active
    camera_active = active
    return f"Camera {'ON' if active else 'OFF'}"

# Google Maps HTML component
def get_maps_html():
    return """
    <!DOCTYPE html>
    <html>
      <head>
        <title>Location Map</title>
        <style>
          .map-container {
            width: 100%;
            height: 100%;
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
          }
          .map-frame {
            width: 100%;
            height: 450px;
            border: none;
            border-radius: 0 0 10px 10px;
          }
          .location-header {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            padding: 15px;
            margin: 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-radius: 10px 10px 0 0;
          }
          .location-info {
            margin: 0;
            font-size: 1.1em;
            font-weight: bold;
          }
          .refresh-button {
            background-color: #fff;
            color: #2980b9;
            border: none;
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
          }
          .refresh-button:hover {
            background-color: #eaf2f8;
            transform: scale(1.05);
          }
          .accuracy-info {
            margin: 0;
            font-size: 0.9em;
            opacity: 0.8;
          }
          .pin-info {
            font-style: italic;
            margin-top: 5px;
            font-size: 0.9em;
          }
        </style>
        <script>
          // Function to get user location and update the map
          function getUserLocation() {
            const infoElement = document.getElementById('location-info');
            const accuracyElement = document.getElementById('accuracy-info');
            
            infoElement.textContent = "Detecting your location...";
            accuracyElement.textContent = "";
            
            if (navigator.geolocation) {
              navigator.geolocation.getCurrentPosition(
                function(position) {
                  const lat = position.coords.latitude;
                  const lng = position.coords.longitude;
                  const accuracy = position.coords.accuracy;
                  const mapFrame = document.getElementById('map-frame');
                  
                  // Update the map iframe with user's coordinates
                  mapFrame.src = `https://www.google.com/maps/embed/v1/place?key=AIzaSyATSRuaTlNp7YG8N7pY8tH0lWvAWGUP6So&q=${lat},${lng}&zoom=15`;
                  
                  // Display coordinates
                  infoElement.textContent = `Your Location: ${lat.toFixed(6)}, ${lng.toFixed(6)}`;
                  if (accuracy) {
                    accuracyElement.textContent = `Accuracy: ${Math.round(accuracy)} meters`;
                  }
                  
                  // Store coordinates in localStorage for persistence
                  localStorage.setItem('userLat', lat);
                  localStorage.setItem('userLng', lng);
                },
                function(error) {
                  console.log("Error getting location:", error);
                  infoElement.textContent = "Could not get your location. Using default location.";
                  
                  // Try to use stored coordinates
                  const storedLat = localStorage.getItem('userLat');
                  const storedLng = localStorage.getItem('userLng');
                  
                  if (storedLat && storedLng) {
                    const mapFrame = document.getElementById('map-frame');
                    mapFrame.src = `https://www.google.com/maps/embed/v1/place?key=AIzaSyATSRuaTlNp7YG8N7pY8tH0lWvAWGUP6So&q=${storedLat},${storedLng}&zoom=15`;
                    infoElement.textContent = `Using last known location: ${storedLat}, ${storedLng}`;
                  }
                }
              );
            } else {
              infoElement.textContent = "Geolocation is not supported by this browser.";
            }
          }
          
          // Initialize when the document is loaded
          document.addEventListener('DOMContentLoaded', function() {
            getUserLocation();
            
            // Add event listener for the refresh button
            document.getElementById('refresh-location').addEventListener('click', function() {
              getUserLocation();
            });
          });
        </script>
      </head>
      <body>
        <div class="map-container">
          <div class="location-header">
            <div>
              <p class="location-info" id="location-info">Detecting your location...</p>
              <p class="accuracy-info" id="accuracy-info"></p>
              <p class="pin-info">📍 Pin indicates exact camera location</p>
            </div>
            <button id="refresh-location" class="refresh-button">Refresh Location</button>
          </div>
          <iframe 
            id="map-frame"
            class="map-frame"
            src="https://www.google.com/maps/embed/v1/place?key=AIzaSyATSRuaTlNp7YG8N7pY8tH0lWvAWGUP6So&q=Your+Location&zoom=10"
            allowfullscreen>
          </iframe>
        </div>
      </body>
    </html>
    """

# Main processing function - this is what we'll connect to Gradio
def process_webcam(video_feed):
    global emergency_count, camera_active
    
    if not camera_active:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Camera Off", (240, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return blank, get_maps_html()
    
    if video_feed is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "No Video Feed", (240, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return blank, get_maps_html()
    
    # Process the frame with YOLOv8 and CLIP
    frame = video_feed.copy()
    
    try:
        # Run detection
        results = yolo_model(frame)[0]
        
        # Reset emergency count for this frame
        emergency_count = 0
        
        # Process each detected object
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Extract region
            try:
                cropped = frame[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue
                
                # Process with CLIP
                image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                image_input = preprocess(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    image_features = clip_model.encode_image(image_input)
                    text_features = clip_model.encode_text(text_prompt)
                    
                    # Calculate similarity with each category
                    similarities = torch.cosine_similarity(image_features, text_features, dim=1)
                    max_sim_value, max_sim_idx = torch.max(similarities, dim=0)
                    max_sim_value = max_sim_value.item()
                    max_sim_category = categories[max_sim_idx]
                
                # If emergency vehicle detected
                if max_sim_value > 0.28:
                    emergency_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = max_sim_category.replace("a ", "").replace("an ", "")
                    cv2.putText(frame, f"{label} ({max_sim_value:.2f})", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            except Exception as e:
                print(f"Error processing detection: {e}")
                continue
        
        # Add emergency count to the frame
        cv2.putText(frame, f"Emergency Vehicles: {emergency_count}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Play alert if emergency vehicles detected
        if emergency_count > 0:
            threading.Thread(target=play_alert, daemon=True).start()
    
    except Exception as e:
        print(f"Error in processing: {e}")
        cv2.putText(frame, "Error in detection", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Return both frame and maps HTML
    return frame, get_maps_html()

# Simplified approach using Interface instead of Blocks
demo = gr.Interface(
    fn=process_webcam,
    inputs=gr.Image(sources=["webcam"], streaming=True, type="numpy"), # Note: 'sources' with an 's'
    outputs=[
        gr.Image(type="numpy", label="Detection Results"),
        gr.HTML(value=get_maps_html(), label="Location")
    ],
    live=True, # Important for real-time processing
    title="🚑 Real-time Emergency Vehicle Detection",
    description="""
    ## Instructions
    - Allow camera access when prompted
    - The system will automatically detect emergency vehicles in real-time
    - Red boxes highlight detected emergency vehicles with their category
    - A count is displayed at the top of the video
    - Audio alert will play when emergency vehicles are detected
    """
)

# Launch the app
if __name__ == "__main__":
    print("Starting real-time emergency vehicle detection")
    demo.queue().launch(share=True)