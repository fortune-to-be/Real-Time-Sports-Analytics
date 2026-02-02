import streamlit as st
import cv2
from ultralytics import YOLO
import torch
from torchvision import transforms
from torchvision.models.video import r3d_18
from PIL import Image
import numpy as np

# Load models
player_model = YOLO("yolov5s.pt")  
try:
    action_model = r3d_18(pretrained=True)  
    action_model.eval()
except Exception as e:
    st.error(f"Failed to load action model: {str(e)}")
    st.stop()

st.title("⚽ Real-Time Sports Analytics System")
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Player Detection Confidence", 0.1, 1.0, 0.5)
source = st.sidebar.selectbox("Video Source", ["Webcam", "Upload Video"])

# Initialize cap
cap = None
if source == "Upload Video":
    uploaded_file = st.file_uploader("Choose video...", type=["mp4", "avi"])
    if uploaded_file:
        with open("temp_video.mp4", 'wb') as f:
            f.write(uploaded_file.read())
        cap = cv2.VideoCapture("temp_video.mp4")
else:
    cap = cv2.VideoCapture(0)

# Check if cap is initialized and opened
if cap is None or not cap.isOpened():
    st.error("Failed to initialize video source. Please upload a valid video file or ensure your webcam is connected.")
    st.stop()

frame_placeholder = st.empty()
stats_placeholder = st.empty()
debug_placeholder = st.empty()  # Placeholder for debug output

# Define transform for action model input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Buffer for action model (ResNet3D-18 expects 16 frames)
frame_buffer = []

# Function to detect dominant jersey color and assign team
def get_team_name(roi):
    try:
        # Validate ROI
        if roi.size == 0 or roi.shape[0] < 1 or roi.shape[1] < 1:
            debug_placeholder.write("Invalid ROI: Empty or too small")
            return "Unknown"
        # Convert ROI to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        # Flatten the ROI to get HSV values
        hsv_flat = hsv_roi.reshape(-1, 3)
        # Compute mean HSV instead of mode for robustness
        mean_hsv = np.mean(hsv_flat, axis=0).astype(int)
        hue, sat, val = mean_hsv
        # Debug: Output HSV values and ROI shape
        debug_placeholder.write(f"ROI shape: {roi.shape}, HSV: Hue={hue}, Sat={sat}, Val={val}")
        # Define color ranges for generic teams
        if 10 <= hue <= 50 and sat > 30 and val > 30:  # Broad yellow range for Team Yellow
            return "Team A"
        elif 100 <= hue <= 140 and sat > 30 and val > 30:  # Blue range for Team Blue
            return "Unkown"
        else:
            return "Team B"
    except Exception as e:
        debug_placeholder.write(f"Team detection error: {str(e)}")
        return "Unknown"

# Action recognition function
def recognize_action(frame):
    try:
        # Convert frame to PIL Image and apply transforms
        img = Image.fromarray(frame)
        img_tensor = transform(img)  # Shape: (3, 224, 224)
        frame_buffer.append(img_tensor)
        if len(frame_buffer) > 16:
            frame_buffer.pop(0)
        if len(frame_buffer) == 16:
            clip = torch.stack(frame_buffer, dim=1)  # Shape: (3, 16, 224, 224)
            clip = clip.unsqueeze(0)  # Shape: (1, 3, 16, 224, 224)
            with torch.no_grad():
                outputs = action_model(clip)
                _, pred = outputs.max(1)
                # Subset of Kinetics-400 labels relevant to sports
                action_labels = ["Idle", "Running", "Kicking Ball", "Tackling"]  # Simplified mapping
                return f"Action: {action_labels[pred.item() % len(action_labels)]}"
        return "Action: Collecting frames..."
    except Exception as e:
        return f"Action: Error ({str(e)})"

# Initialize stats for teams
team_counts = {"Team A": 0, "Unknown": 0, "Team B": 0}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Player detection
    results = player_model(rgb_frame)
    team_counts = {"Team A": 0, "Unknown": 0, "Team B": 0}  # Reset counts per frame
    for box in results[0].boxes:
        if box.conf > confidence_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Extract ROI (full bounding box for more jersey coverage)
            roi = rgb_frame[y1:y2, x1:x2]
            if roi.size == 0:  # Skip empty ROIs
                debug_placeholder.write("Skipped empty ROI")
                continue
            team_name = get_team_name(roi)
            team_counts[team_name] += 1
            # Draw bounding box and label with team name
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Player ({team_name})"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    # Action recognition
    action_label = recognize_action(rgb_frame)
    cv2.putText(frame, action_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    frame_placeholder.image(frame, channels="BGR")
    # Update stats
    stats_placeholder.markdown(
        f"**Players Detected:** {len(results[0].boxes)}<br>"
        f"**Team A:** {team_counts['Team A']}<br>"
        f"**Unknown:** {team_counts['Unknown']}<br>"
        f"**Team B:** {team_counts['Team B']}<br>"
        f"**{action_label}**",
        unsafe_allow_html=True
    )
    if source == "Webcam":
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()