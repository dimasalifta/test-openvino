from ultralytics import YOLO
import cv2
import time
# Load a YOLOv8n PyTorch model
model = YOLO("yolo11m.pt")

# Export the model
model.export(format="openvino", opset=11)  # creates 'yolov8n_openvino_model/'

# Load the exported OpenVINO model
ov_model = YOLO("yolo11m_openvino_model/")

# Run inference
results = ov_model("https://ultralytics.com/images/bus.jpg")
# Buka video (ganti path jika perlu)
video_path = "input.mp4"
cap = cv2.VideoCapture(0)
# Inisialisasi FPS tracker
prev_time = time.time()
# Loop proses frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Hitung waktu mulai untuk FPS
    start_time = time.time()

    # Inference dengan device (jika didukung)
    results = ov_model(frame, device="intel:npu")
    
    # print(results)
    # Ambil hasil deteksi dari frame (YOLO kadang mengembalikan list of results)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Ambil koordinat dan kelas
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = r.names[cls_id]

            # Gambar bounding box dan label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #  Hitung FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # Tampilkan FPS di frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # Tampilkan frame
    cv2.imshow("YOLOv8 OpenVINO Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
