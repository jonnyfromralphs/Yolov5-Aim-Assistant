import cv2 as cv
import torch
from mss import mss
import numpy as np
from PIL import Image
import time
import pyautogui
import pydirectinput as pi
import keyboard as kb

# load yolov5 with custom trained model. model trained on csgo images to predict images of two classes: player bodies and heads
model = torch.hub.load("C:/Users/jonny/Desktop/Side Projects/yolov5-master", "custom",
 path="C:/Users/jonny/Desktop/Side Projects/Object Detection/csgo.pt", source="local", force_reload=True)
# set capture coordinates and size - middle of screen 400 x 400px
captureSize = {'top': 340, 'left': 760, 'width': 400, 'height': 400}
# use mss to screenshot portion of screen
sct = mss()

# move mouse function placeholder for better mouse input method
def moveMouse(x, y):
    pi.moveTo(x, y)

#min threshold value for detections
MIN_THRESHOLD = 0.4

while True:
    t = time.time()
    # grab image of captureSize
    img = sct.grab(captureSize)
    # convert img to np array to pass into model
    screen = np.array(img)
    # get results from passing in np screen array
    result = model(cv.cvtColor(screen, cv.COLOR_BGR2RGB), size=400)
    detected = np.squeeze(result.render())
    # convert predictions into list
    tensorList = result.xyxy[0].tolist()
    # if len of the list is greater than 0, possible enemy is on screen within bounding box
    if len(tensorList) > 0:
        # gets confidence score
        confidence = tensorList[0][4]
        # gets detected image class - body or head
        detectedImage = tensorList[0][5]
        # if confidence score is greater than min threshold
        if confidence > MIN_THRESHOLD:
            if detectedImage == 1 or detectedImage == 0:
                # get x and y coordinates of bounding box and get ~middle point
                xMin = int(tensorList[0][0])
                xMax = int(tensorList[0][2])
                yMin = int(tensorList[0][1])
                yMax = int(tensorList[0][3])
                middleX = (xMax + xMin) // 2
                middleY = (yMax + yMin) // 2
                # add to x and y coordinates for a 1920x1080 screen given calc coordinates
                x = middleX + 760
                y = yMin + 350
                # uses opencv to draw line to detected image
                cv.line(detected, (200, 200), (middleX, middleY), (255, 0, 0), 1)
                # if z is pressed, mouse will move to coordinates
                if kb.is_pressed("z"):
                    moveMouse(x, y)

    # calculate fps using time library and format into text
    fps = 1 / (time.time() - t)
    fps_text = "FPS: {:.0f}".format(fps)
    winName = "Computer Vision"
    winTitle = winName + " FPS: {}".format(fps_text)
    # overlay fps on window shown by open cv
    cv.putText(detected, fps_text, (10,20), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1, cv.LINE_AA)
    cv.imshow(winName, detected)

    #quit loop by pressing q
    if cv.waitKey(25) & 0xFF == ord("q"):
        cv.destroyAllWindows()
        break

