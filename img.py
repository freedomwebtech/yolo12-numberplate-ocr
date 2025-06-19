import cv2
import time
import os

# === Settings ===
output_folder = "images"
maxFrames = 200
delay = 0.1  # seconds

# === Create folder if it doesn't exist ===
os.makedirs(output_folder, exist_ok=True)

# === Initialize USB camera (typically index 0) ===
cap = cv2.VideoCapture('vid1.mp4')
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

cpt = 0

while cpt < maxFrames:
    ret, frame = cap.read()
    frame=cv2.resize(frame,(1020,600))
    if not ret:
        print("Failed to capture frame from camera.")
        break

    filename = os.path.join(output_folder, f"Numberplate_{cpt}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Saved {filename}")

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC key to stop
        break

    cpt += 1
    time.sleep(delay)

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
