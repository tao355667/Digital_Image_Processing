import numpy as np
import cv2 as cv
# mouse callback function
# typedef void(*cv::MouseCallback)(int event, int x, int y, int flags, void * userdata)


def draw_circle(event, x, y, flag, param):
    # if event == cv.EVENT_LBUTTONDOWN:
    if flag == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img, (x, y), 10, (0, 255, 255), -1)
        


# Create a black image, a window and bind the function to window
img = np.zeros((512, 512, 3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image', draw_circle)
while (1):
    cv.imshow('image', img)
    if cv.waitKey(20) & 0xFF == 27:
        break
cv.destroyAllWindows()
