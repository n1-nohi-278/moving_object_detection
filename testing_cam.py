import cv2  # opencv

vs = cv2.VideoCapture(0)  # initialize the camera

while True:
    _, img = vs.read()  # reading the frame from the camera

    cv2.imshow("VideoStream", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

vs.release()  # releasing the camera
cv2.destroyAllWindows()  # close all windows