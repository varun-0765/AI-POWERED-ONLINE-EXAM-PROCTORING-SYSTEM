import cv2
from ultralytics import YOLO

# Load the model - 'yolov8n' is fast, 'yolov8s' is more accurate for phones
model = YOLO('yolov8n.pt') 

cap = cv2.VideoCapture(0)

# Persistence counters to make warnings "smooth"
head_warn_frames = 0
phone_warn_frames = 0
FRAME_BUFFER = 10 # Alert only if violation lasts ~0.5 seconds

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    screen_center_x = w / 2
    
    # 1. Detection with lower confidence to catch small phones
    results = model(frame, conf=0.25, verbose=False)
    
    phone_in_frame = False
    head_deviation = False
    gaze_text = "CENTERED"
    gaze_color = (0, 255, 0)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            # --- HEAD POSITION LOGIC ---
            if label == "person":
                b = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, b)
                
                # Find the center of the person's head
                head_x = (x1 + x2) / 2
                
                # Check if head moved too far left or right (30% deviation)
                if head_x < screen_center_x - (w * 0.15):
                    head_deviation = True
                    gaze_text, gaze_color = "WARNING: MOVED LEFT", (0, 255, 255)
                elif head_x > screen_center_x + (w * 0.15):
                    head_deviation = True
                    gaze_text, gaze_color = "WARNING: MOVED RIGHT", (0, 255, 255)
                
                # Draw subtle candidate box
                cv2.rectangle(frame, (x1, y1), (x2, y2), gaze_color, 1)

            # --- PHONE DETECTION LOGIC ---
            if label == "cell phone":
                phone_in_frame = True
                b = box.xyxy[0].cpu().numpy()
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)

    # 2. SMOOTHING / BUFFER LOGIC
    if head_deviation: head_warn_frames += 1
    else: head_warn_frames = 0

    if phone_in_frame: phone_warn_frames += 1
    else: phone_warn_frames = 0

    # 3. PROFESSIONAL HUD (Heads-Up Display)
    # Top Bar Overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w, 50), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "PROCTOR LIVE FEED", (20, 32), 0, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, gaze_text, (w - 300, 32), 0, 0.6, gaze_color, 2)

    # Trigger Heavy Warnings
    if head_warn_frames > FRAME_BUFFER:
        cv2.putText(frame, "STATUS: UNUSUAL HEAD MOVEMENT", (20, h - 30), 0, 0.8, (0, 0, 255), 2)
    
    if phone_warn_frames > 3: # Phones trigger almost instantly
        cv2.putText(frame, "STATUS: UNAUTHORIZED DEVICE", (w - 380, h - 30), 0, 0.8, (0, 0, 255), 2)

    cv2.imshow('AI Proctor Professional v4', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()