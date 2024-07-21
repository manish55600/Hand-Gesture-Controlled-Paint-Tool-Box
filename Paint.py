import tkinter as tk
from tkinter import Message ,Text
import tkinter.ttk as ttk
import tkinter.font as font
from PIL import Image,ImageTk
from tkinter import *

window = tk.Tk()

window.title("Face_Recogniser")
window.geometry("1400x800")

"""photo = PhotoImage(file = "paint_sym.png")
w = Label(root, image=photo)
w.pack()"""

filename = PhotoImage(file = "F:/PAINT_TOOOL/complete_code/paint_sym1.png")
background_label = Label(window, image=filename)
background_label.place( relwidth=1, relheight=1)
"""
background_image=tk.PhotoImage()
background_label = tk.Label(parent, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
dialog_title = 'QUIT'
dialog_text = 'Are you sure?'
"""
window.configure(background='white')

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

message = tk.Label(window, text="_"*9+"Real Time Paint Tool Box  Hand Gestures"+"_"*9, bg="black", fg="white", width=100, height=2,
                    font=("Tempus Sans ITC",19,"bold"))

message.place(x=0, y=0)


def rectangle():
    import cv2
    import numpy as np
    import math

    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        # read image
        ret, img = cap.read()

        # get hand data from the rectangle sub window on the screen
        cv2.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 0)
        crop_img = img[100:300, 100:300]

        # convert to grayscale
        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        # applying gaussian blur
        value = (35, 35)
        blurred = cv2.GaussianBlur(grey, value, 0)

        # thresholdin: Otsu's Binarization method
        _, thresh1 = cv2.threshold(blurred, 127, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # show thresholded image
        cv2.imshow('Thresholded', thresh1)

        # check OpenCV version to avoid unpacking error
        (version, _, _) = cv2.__version__.split('.')

        if version == '3':
            image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
                                                          cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        elif version == '2':
            contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, \
                                                   cv2.CHAIN_APPROX_NONE)

        # find contour with max area
        cnt = max(contours, key=lambda x: cv2.contourArea(x))

        # create bounding rectangle around the contour (can skip below two lines)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # finding convex hull
        hull = cv2.convexHull(cnt)

        # drawing contours
        drawing = np.zeros(crop_img.shape, np.uint8)
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)

        # finding convex hull
        hull = cv2.convexHull(cnt, returnPoints=False)

        # finding convexity defects
        defects = cv2.convexityDefects(cnt, hull)
        count_defects = 0
        cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

        # applying Cosine Rule to find angle for all defects (between fingers)
        # with angle > 90 degrees and ignore defects
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]

            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

            # apply cosine rule here
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            # ignore angles > 90 and highlight rest with red dots
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_img, far, 1, [0, 0, 255], -1)
            # dist = cv2.pointPolygonTest(cnt,far,True)

            # draw a line from start to end i.e. the convex points (finger tips)
            # (can skip this part)
            cv2.line(crop_img, start, end, [0, 255, 0], 2)
            # cv2.circle(crop_img,far,5,[0,0,255],-1)

        if count_defects == 1:

            import turtle

            t = turtle.Turtle()
            t.forward(100)
            t.left(90)
            t.forward(100)
            t.left(90)
            t.forward(100)
            t.left(90)
            t.forward(100)
            t.left(90)

        elif count_defects == 2:

            import turtle

            t = turtle.Turtle()

            for i in range(5):
                t.forward(150)
                t.right(144)

            cv2.putText(img, str, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        elif count_defects == 3:
            cv2.putText(img, "This is 4 :P", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 4:
            cv2.putText(img, "Hi!!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        else:
            cv2.putText(img, "show rectangle!!!", (50, 50), \
                        cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

        # show appropriate images in windows
        cv2.imshow('Gesture', img)
        all_img = np.hstack((drawing, crop_img))
        cv2.imshow('Contours', all_img)

        k = cv2.waitKey(10)
        if k == 27:
            break

    import turtle

    t = turtle.Turtle()
    t.forward(100)
    t.left(90)
    t.forward(100)
    t.left(90)
    t.forward(100)
    t.left(90)
    t.forward(100)
    t.left(90)


def star():
    import cv2
    import numpy as np
    import math

    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        # read image
        ret, img = cap.read()

        # get hand data from the rectangle sub window on the screen
        cv2.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 0)
        crop_img = img[100:300, 100:300]

        # convert to grayscale
        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        # applying gaussian blur
        value = (35, 35)
        blurred = cv2.GaussianBlur(grey, value, 0)

        # thresholdin: Otsu's Binarization method
        _, thresh1 = cv2.threshold(blurred, 127, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # show thresholded image
        cv2.imshow('Thresholded', thresh1)

        # check OpenCV version to avoid unpacking error
        (version, _, _) = cv2.__version__.split('.')

        if version == '3':
            image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
                                                          cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        elif version == '2':
            contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, \
                                                   cv2.CHAIN_APPROX_NONE)

        # find contour with max area
        cnt = max(contours, key=lambda x: cv2.contourArea(x))

        # create bounding rectangle around the contour (can skip below two lines)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # finding convex hull
        hull = cv2.convexHull(cnt)

        # drawing contours
        drawing = np.zeros(crop_img.shape, np.uint8)
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)

        # finding convex hull
        hull = cv2.convexHull(cnt, returnPoints=False)

        # finding convexity defects
        defects = cv2.convexityDefects(cnt, hull)
        count_defects = 0
        cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

        # applying Cosine Rule to find angle for all defects (between fingers)
        # with angle > 90 degrees and ignore defects
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]

            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

            # apply cosine rule here
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            # ignore angles > 90 and highlight rest with red dots
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_img, far, 1, [0, 0, 255], -1)
            # dist = cv2.pointPolygonTest(cnt,far,True)

            # draw a line from start to end i.e. the convex points (finger tips)
            # (can skip this part)
            cv2.line(crop_img, start, end, [0, 255, 0], 2)
            # cv2.circle(crop_img,far,5,[0,0,255],-1)

        # define actions required
        if count_defects == 1:
            import turtle

            star = turtle.Turtle()

            for i in range(50):
                star.forward(50)
                star.right(144)

            turtle.done()

        elif count_defects == 2:

            import turtle

            t = turtle.Turtle()

            for i in range(5):
                t.forward(150)
                t.right(144)
            turtle.done()
            cv2.putText(img, str, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        elif count_defects == 3:
            cv2.putText(img, "This is 4 :P", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 4:
            cv2.putText(img, "Hi!!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        else:
            cv2.putText(img, "show star!!!", (50, 50), \
                        cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

        # show appropriate images in windows
        cv2.imshow('Gesture', img)
        all_img = np.hstack((drawing, crop_img))
        cv2.imshow('Contours', all_img)

        k = cv2.waitKey(10)
        if k == 27:
            break

    import turtle

    t = turtle.Turtle()

    for i in range(5):
        t.forward(150)
        t.right(144)




def triangle():
    import cv2
    import numpy as np
    import math

    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        # read image
        ret, img = cap.read()

        # get hand data from the rectangle sub window on the screen
        cv2.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 0)
        crop_img = img[100:300, 100:300]

        # convert to grayscale
        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        # applying gaussian blur
        value = (35, 35)
        blurred = cv2.GaussianBlur(grey, value, 0)

        # thresholdin: Otsu's Binarization method
        _, thresh1 = cv2.threshold(blurred, 127, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # show thresholded image
        cv2.imshow('Thresholded', thresh1)

        # check OpenCV version to avoid unpacking error
        (version, _, _) = cv2.__version__.split('.')

        if version == '3':
            image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
                                                          cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        elif version == '2':
            contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, \
                                                   cv2.CHAIN_APPROX_NONE)

        # find contour with max area
        cnt = max(contours, key=lambda x: cv2.contourArea(x))

        # create bounding rectangle around the contour (can skip below two lines)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # finding convex hull
        hull = cv2.convexHull(cnt)

        # drawing contours
        drawing = np.zeros(crop_img.shape, np.uint8)
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)

        # finding convex hull
        hull = cv2.convexHull(cnt, returnPoints=False)

        # finding convexity defects
        defects = cv2.convexityDefects(cnt, hull)
        count_defects = 0
        cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

        # applying Cosine Rule to find angle for all defects (between fingers)
        # with angle > 90 degrees and ignore defects
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]

            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

            # apply cosine rule here
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            # ignore angles > 90 and highlight rest with red dots
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_img, far, 1, [0, 0, 255], -1)
            # dist = cv2.pointPolygonTest(cnt,far,True)

            # draw a line from start to end i.e. the convex points (finger tips)
            # (can skip this part)
            cv2.line(crop_img, start, end, [0, 255, 0], 2)
            # cv2.circle(crop_img,far,5,[0,0,255],-1)

        if count_defects == 1:

            import turtle

            board = turtle.Turtle()
            board.forward(100)  # draw base
            board.left(120)
            board.forward(100)
            board.left(120)
            board.forward(100)
            turtle.done()
        elif count_defects == 2:

            import turtle

            t = turtle.Turtle()

            for i in range(5):
                t.forward(150)
                t.right(144)

            cv2.putText(img, str, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        elif count_defects == 3:
            cv2.putText(img, "This is 4 :P", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 4:
            cv2.putText(img, "Hi!!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        else:
            cv2.putText(img, "show triangle!!!", (50, 50), \
                        cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

        # show appropriate images in windows
        cv2.imshow('Gesture', img)
        all_img = np.hstack((drawing, crop_img))
        cv2.imshow('Contours', all_img)

        k = cv2.waitKey(10)
        if k == 27:
            break

    import turtle

    t = turtle.Turtle()
    t.forward(100)
    t.left(90)
    t.forward(100)
    t.left(90)
    t.forward(100)
    t.left(90)
    t.forward(100)
    t.left(90)


def colour():
    import numpy as np
    import cv2
    from collections import deque

    # Define the upper and lower boundaries for a color to be considered "Blue"
    blueLower = np.array([100, 60, 60])
    blueUpper = np.array([140, 255, 255])
    #skin =np.array([231,158,109])

    # Define a 5x5 kernel for erosion and dilation
    kernel = np.ones((5, 5), np.uint8)

    # Setup deques to store separate colors in separate arrays
    bpoints = [deque(maxlen=512)]
    gpoints = [deque(maxlen=512)]
    rpoints = [deque(maxlen=512)]
    ypoints = [deque(maxlen=512)]
    opoints = [deque(maxlen=512)]
    vpoints = [deque(maxlen=512)]
    ipoints = [deque(maxlen=512)]
    bindex = 0
    gindex = 0
    rindex = 0
    yindex = 0
    oindex = 0
    vindex = 0
    iindex = 0

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255),(255, 165 , 0),(238 , 130, 238),(75,0 ,130)]

    colorIndex = 0

    # Setup the Paint interface
    paintWindow = np.zeros((471, 636, 3)) + 255
    paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
    paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), colors[0], -1)
    paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), colors[1], -1)
    paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), colors[2], -1)
    paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), colors[3], -1)
    paintWindow = cv2.rectangle(paintWindow, (620, 1), (715, 65), colors[4], -1)
    paintWindow = cv2.rectangle(paintWindow, (735, 1), (830, 65), colors[5], -1)
    paintWindow = cv2.rectangle(paintWindow, (850, 1), (945, 65), colors[6], -1)
    cv2.putText(paintWindow, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "ORANGE", (640,33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165 , 0),2,cv2.LINE_AA)
    cv2.putText(paintWindow, "VIOOLET", (762, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "INDIGO", (882, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2, cv2.LINE_AA)
    cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

    # Load the video
    camera = cv2.VideoCapture(0)

    # Keep looping
    while True:
        # Grab the current paintWindow
        (grabbed, frame) = camera.read()
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Add the coloring options to the frame
        frame = cv2.rectangle(frame, (40, 1), (140, 65), (122, 122, 122), -1)
        frame = cv2.rectangle(frame, (160, 1), (255, 65), colors[0], -1)
        frame = cv2.rectangle(frame, (275, 1), (370, 65), colors[1], -1)
        frame = cv2.rectangle(frame, (390, 1), (485, 65), colors[2], -1)
        frame = cv2.rectangle(frame, (505, 1), (600, 65), colors[3], -1)
        frame = cv2.rectangle(frame, (620, 1), (715, 65), colors[4], -1)
        frame = cv2.rectangle(frame, (735, 1), (830, 65), colors[5], -1)
        frame = cv2.rectangle(frame, (850, 1), (945, 65), colors[6], -1)
        cv2.putText(frame, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)
        cv2.putText(frame, "ORANGE", (640, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 153, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "VIOLET", (762, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 153, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "INDIGO", (882, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 153, 255), 2, cv2.LINE_AA)
        # Check to see if we have reached the end of the video
        if not grabbed:
            break

        # Determine which pixels fall within the blue boundaries and then blur the binary image
        blueMask = cv2.inRange(hsv, blueLower,blueUpper)
        blueMask = cv2.erode(blueMask, kernel, iterations=2)
        blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
        blueMask = cv2.dilate(blueMask, kernel, iterations=1)

        # Find contours in the image
        (_, cnts, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        center = None

        # Check to see if any contours were found
        if len(cnts) > 0:
            # Sort the contours and find the largest one -- we
            # will assume this contour correspondes to the area of the bottle cap
            cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
            # Get the radius of the enclosing circle around the found contour
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            # Draw the circle around the contour
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            # Get the moments to calculate the center of the contour (in this case Circle)
            M = cv2.moments(cnt)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

            if center[1] <= 65:
                if 40 <= center[0] <= 140:  # Clear All
                    bpoints = [deque(maxlen=512)]
                    gpoints = [deque(maxlen=512)]
                    rpoints = [deque(maxlen=512)]
                    ypoints = [deque(maxlen=512)]
                    opoints = [deque(maxlen=512)]
                    vpoints = [deque(maxlen=512)]
                    ipoints = [deque(maxlen=512)]
                    bindex = 0
                    gindex = 0
                    rindex = 0
                    yindex = 0
                    oindex = 0
                    vindex = 0
                    iindex = 0


                    paintWindow[67:, :, :] = 255
                elif 160 <= center[0] <= 255:
                    colorIndex = 0  # Blue
                elif 275 <= center[0] <= 370:
                    colorIndex = 1  # Green
                elif 390 <= center[0] <= 485:
                    colorIndex = 2  # Red
                elif 505 <= center[0] <= 600:
                    colorIndex = 3  # Yellow
                elif 620 <= center[0] <= 715:
                    colorIndex = 3  # Orange
                elif 735 <= center[0] <= 830:
                    colorIndex = 3  # Violet
                elif 850 <= center[0] <= 945:
                    colorIndex = 3  # Indigo
            else:
                if colorIndex == 0:
                    bpoints[bindex].appendleft(center)
                elif colorIndex == 1:
                    gpoints[gindex].appendleft(center)
                elif colorIndex == 2:
                    rpoints[rindex].appendleft(center)
                elif colorIndex == 3:
                    ypoints[yindex].appendleft(center)
                elif colorIndex ==4:
                    opoints[oindex].apppendleft(center)
                elif colorIndex ==5:
                    vpoints[vindex].apppendleft(center)
                elif colorIndex == 6:
                    ipoints[iindex].apppendleft(center)
        # Append the next deque when no contours are detected (i.e., bottle cap reversed)
        else:
            bpoints.append(deque(maxlen=512))
            bindex += 1
            gpoints.append(deque(maxlen=512))
            gindex += 1
            rpoints.append(deque(maxlen=512))
            rindex += 1
            ypoints.append(deque(maxlen=512))
            yindex += 1
            opoints.append(deque(maxlen=512))
            oindex += 1
            vpoints.append(deque(maxlen=512))
            vindex += 1
            ipoints.append(deque(maxlen=512))
            iindex += 1

        # Draw lines of all the colors (Blue, Green, Red and Yellow)
        points = [bpoints, gpoints, rpoints, ypoints,opoints,vpoints,ipoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

        # Show the frame and the paintWindow image
        cv2.imshow("Tracking", frame)
        cv2.imshow("Paint", paintWindow)

        # If the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()

def crop_image():
    import cv2
    import numpy as np
    import math

    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        # read image
        ret, img = cap.read()

        # get hand data from the rectangle sub window on the screen
        cv2.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 0)
        crop_img = img[100:300, 100:300]

        # convert to grayscale
        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        # applying gaussian blur
        value = (35, 35)
        blurred = cv2.GaussianBlur(grey, value, 0)

        # thresholdin: Otsu's Binarization method
        _, thresh1 = cv2.threshold(blurred, 127, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # show thresholded image
        cv2.imshow('Thresholded', thresh1)

        # check OpenCV version to avoid unpacking error
        (version, _, _) = cv2.__version__.split('.')

        if version == '3':
            image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
                                                          cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        elif version == '2':
            contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, \
                                                   cv2.CHAIN_APPROX_NONE)

        # find contour with max area
        cnt = max(contours, key=lambda x: cv2.contourArea(x))

        # create bounding rectangle around the contour (can skip below two lines)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # finding convex hull
        hull = cv2.convexHull(cnt)

        # drawing contours
        drawing = np.zeros(crop_img.shape, np.uint8)
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)

        # finding convex hull
        hull = cv2.convexHull(cnt, returnPoints=False)

        # finding convexity defects
        defects = cv2.convexityDefects(cnt, hull)
        count_defects = 0
        cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

        # applying Cosine Rule to find angle for all defects (between fingers)
        # with angle > 90 degrees and ignore defects
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]

            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

            # apply cosine rule here
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            # ignore angles > 90 and highlight rest with red dots
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_img, far, 1, [0, 0, 255], -1)
            # dist = cv2.pointPolygonTest(cnt,far,True)

            # draw a line from start to end i.e. the convex points (finger tips)
            # (can skip this part)
            cv2.line(crop_img, start, end, [0, 255, 0], 2)
            # cv2.circle(crop_img,far,5,[0,0,255],-1)

        # define actions required
        if count_defects == 1:

            from PIL import Image

            def crop(image_path, coords, saved_location):

                image_obj = Image.open(image_path)
                cropped_image = image_obj.crop(coords)
                cropped_image.save(saved_location)
                cropped_image.show()

            if __name__ == '__main__':
                image = 'OriginalImage.png'
                crop(image, (161, 166, 706, 1050), 'cropped.png')


        elif count_defects == 2:

            import turtle

            t = turtle.Turtle()
            t.forward(100)
            t.left(90)
            t.forward(100)
            t.left(90)
            t.forward(100)
            t.left(90)
            t.forward(100)
            t.left(90)

            cv2.putText(img, str, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        elif count_defects == 3:
            cv2.putText(img, "This is 4 :P", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 4:
            cv2.putText(img, "Hi!!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        else:
            cv2.putText(img, "crop image!!!", (50, 50), \
                        cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

        # show appropriate images in windows
        cv2.imshow('Gesture', img)
        all_img = np.hstack((drawing, crop_img))
        cv2.imshow('Contours', all_img)

        k = cv2.waitKey(10)
        if k == 27:
            break

from turtle import *

def Circle():
    import cv2
    import numpy as np
    import math

    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        # read image
        ret, img = cap.read()

        # get hand data from the rectangle sub window on the screen
        cv2.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 0)
        crop_img = img[100:300, 100:300]

        # convert to grayscale
        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        # applying gaussian blur
        value = (35, 35)
        blurred = cv2.GaussianBlur(grey, value, 0)

        # thresholdin: Otsu's Binarization method
        _, thresh1 = cv2.threshold(blurred, 127, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # show thresholded image
        cv2.imshow('Thresholded', thresh1)

        # check OpenCV version to avoid unpacking error
        (version, _, _) = cv2.__version__.split('.')

        if version == '3':
            image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
                                                          cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        elif version == '2':
            contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, \
                                                   cv2.CHAIN_APPROX_NONE)

        # find contour with max area
        cnt = max(contours, key=lambda x: cv2.contourArea(x))

        # create bounding rectangle around the contour (can skip below two lines)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # finding convex hull
        hull = cv2.convexHull(cnt)

        # drawing contours
        drawing = np.zeros(crop_img.shape, np.uint8)
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)

        # finding convex hull
        hull = cv2.convexHull(cnt, returnPoints=False)

        # finding convexity defects
        defects = cv2.convexityDefects(cnt, hull)
        count_defects = 0
        cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

        # applying Cosine Rule to find angle for all defects (between fingers)
        # with angle > 90 degrees and ignore defects
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]

            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

            # apply cosine rule here
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            # ignore angles > 90 and highlight rest with red dots
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_img, far, 1, [0, 0, 255], -1)
            # dist = cv2.pointPolygonTest(cnt,far,True)

            # draw a line from start to end i.e. the convex points (finger tips)
            # (can skip this part)
            cv2.line(crop_img, start, end, [0, 255, 0], 2)
            # cv2.circle(crop_img,far,5,[0,0,255],-1)

        if count_defects == 1:

            #from turtle import *
            import turtle
            import math

            apple = Turtle()

            def polygon(t, n, length):
                for i in range(n):
                    left(360 / n)
                    forward(length)

            def draw_circle(t, r):
                circumference = 2 * math.pi * r
                n = 50
                length = circumference / n
                polygon(t, n, length)
                exitonclick()

            draw_circle(apple, 30)
        elif count_defects == 2:

            import turtle

            t = turtle.Turtle()

            for i in range(5):
                t.forward(150)
                t.right(144)

            cv2.putText(img, str, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        elif count_defects == 3:
            cv2.putText(img, "This is 4 :P", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 4:
            cv2.putText(img, "Hi!!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        else:
            cv2.putText(img, "show circle!!!", (50, 50), \
                        cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

        # show appropriate images in windows
        cv2.imshow('Gesture', img)
        all_img = np.hstack((drawing, crop_img))
        cv2.imshow('Contours', all_img)

        k = cv2.waitKey(10)
        if k == 27:
            break

def Write():
    from subprocess import call
    call(["python", "Write.py"])

Shape = tk.Label(window, text="--SHAPES--", fg="black", bg="cyan", width=20, height=2,
                     font=("Tempus Sans ITC",15,"bold"))
Shape.place(x=130, y=200)



rectangle = tk.Button(window, text="Rectangle", command=rectangle, fg="black", bg="cyan", width=20, height=2,
                     font=("Tempus Sans ITC",15,"bold"))
rectangle.place(x=130, y=300)

star = tk.Button(window, text="Star", command=star, fg="black", bg="cyan", width=20, height=2,
                     font=("Tempus Sans ITC",15,"bold"))
star.place(x=130, y=390)

triangle = tk.Button(window, text="Triangle", command=triangle, fg="black",bg="cyan", width=20, height=2,
                     font=("Tempus Sans ITC",15,"bold"))
triangle.place(x=130, y=490)


circle = tk.Button(window, text="Circle",command=Circle,fg="black", bg="cyan", width=20, height=2,
                     font=("Tempus Sans ITC",15,"bold"))
circle.place(x=130, y=590)



Ops = tk.Label(window, text="--OPERATIONS--", fg="black", bg="green yellow", width=20, height=2,
                     font=("Tempus Sans ITC",17,"bold"))
Ops.place(x=965, y=200)

colour = tk.Button(window, text="Colour", command=colour, fg="black", bg="green yellow", width=20, height=2,
                     font=("Tempus Sans ITC",15,"bold"))
colour.place(x=1000, y=300)

crop_image = tk.Button(window, text="Crop Image", command=crop_image, fg="black", bg="green yellow", width=20, height=2,
                     font=("Tempus Sans ITC",15,"bold"))
crop_image.place(x=1000, y=400)

Write = tk.Button(window, text="Write", command=Write, fg="black", bg="green yellow", width=20, height=2,
                     font=("Tempus Sans ITC",15,"bold"))
Write.place(x=1000, y=500)



window.mainloop()