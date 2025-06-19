import cv2
from ultralytics import YOLO
import cvzone
from paddleocr import PaddleOCR

# Load Models
model = YOLO('best.pt')  # Assumes this model detects number plates
names = model.names
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Mouse position debug
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

cap = cv2.VideoCapture("vid.mp4")
frame_count = 0
hist = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 600))
    results = model.track(frame, persist=True)

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for track_id, box, class_id in zip(ids, boxes, class_ids):
            x1, y1, x2, y2 = box
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2
            c = names[class_id]

            # Draw rectangle and labels
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cvzone.putTextRect(frame, f'{c.upper()}', (x1, y1 - 10), scale=1, thickness=2,
                               colorT=(255, 255, 255), colorR=(0, 0, 255), offset=5, border=2)
            cvzone.putTextRect(frame, f'ID: {track_id}', (x2, y2 + 10), scale=1, thickness=2,
                               colorT=(255, 255, 255), colorR=(0, 255, 0), offset=5, border=2)

            # Only process number plates
            if c.lower() == "numberplate":
                cropped_plate = frame[y1:y2, x1:x2]
                if cropped_plate.size == 0:
                    continue

                # OCR
                result = ocr.ocr(cropped_plate, cls=True)
                text = ""
                for line in result:
                    if line:
                        text += ''.join([word_info[1][0] for word_info in line]) + ' '

                if text.strip():
                    cvzone.putTextRect(frame, f'{text.strip()}', (x1, y1 - 40), scale=1, thickness=2,
                                       colorT=(0, 0, 0), colorR=(255, 255, 0), offset=5, border=2)

    cv2.imshow("RGB", frame)

    if cv2.waitKey(0) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
