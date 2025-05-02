import cv2

# Buka kamera
cap = cv2.VideoCapture(0)

# Set format ke MJPEG (biar support resolusi tinggi dan 60FPS)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Set resolusi 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Set FPS ke 60
cap.set(cv2.CAP_PROP_FPS, 60)

# Tampilkan properti yang sebenarnya digunakan
print("Resolusi:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FPS:", cap.get(cv2.CAP_PROP_FPS))

# Jalankan loop untuk tampilkan video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

