import cv2

camera = cv2.VideoCapture(0)

print("Camera opened:", camera.isOpened())

while True:

    ret, frame = camera.read()

    print("Frame received:", ret)

    if not ret:
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()