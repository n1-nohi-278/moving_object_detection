import time
import cv2  # image
import imutils  # resize

# Initialize camera and allow it to warm up
cam = cv2.VideoCapture(0)  # cam id (0 is standard for default webcam)
time.sleep(1)

firstFrame = None
area = 500

while True:
    _, img = cam.read()  # read frame from camera
    text = "Normal"
    img = imutils.resize(img, width=500)  # resize for faster processing

    # Preprocessing
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # color to Gray scale image
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)  # smoothened

    # Capture the background baseline
    if firstFrame is None:
        firstFrame = gaussianImg  # capturing 1st frame on 1st iteration
        continue

    # Calculate differences
    imgDiff = cv2.absdiff(firstFrame, gaussianImg)  # absolute diff b/w 1st and current frame
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]  # binary threshold
    threshImg = cv2.dilate(threshImg, None, iterations=2)

    # Find and filter movement contours
    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        
        # If movement is large enough, draw a bounding box
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Moving Object detected"

    # Display results
    print(text)
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("cameraFeed", img)
    
    # Key binding to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()