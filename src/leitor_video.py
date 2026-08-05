import cv2

video = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = video.read()

    if not ret:
        break

    cv2.imshow("Frame", frame)

    if cv2.waitKey(30) == 27:
        break

video.release()
cv2.destroyAllWindows()