import os
import tkinter as tk
from tkinter import ttk, LEFT, END
from tkinter.filedialog import askopenfilename
from PIL import Image
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as spm
import sys
from METHODS import *

sys.path.append('../LIBRARY')

import numpy as np
root = tk.Tk()
root.geometry("1300x700")

lbl = tk.Label(root, text="real time paint tool by hand gestures", font=('times', 20,' bold '), height=1, width=30)
lbl.place(x=430, y=5)

frame_alpr = tk.LabelFrame(root, text=" SHAPE ", width=280, height=600, bd=5, font=('times', 15, ' bold '))
frame_alpr.grid(row=0, column=0, sticky='nw')
frame_alpr.place(x=5, y=50)

def img_input():
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
def preprocessing():
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

            from turtle import *
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


def edges():
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
            cv2.putText(img, "show circle!!!", (50, 50), \
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


def lic_plate_loc():
    # import edges o/p as i/p
    global eg
    edges = eg

    # import orignal image
    global fn
    FName = fn
    imgpath = FName

    img = cv2.imread(imgpath)

    x1 = int(img.shape[0]/3)
    y1 = int(img.shape[1]/3)

    img = cv2.resize(img, (x1, y1))

    (cnts, new) = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCnt = None

    count = 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            break

    cv2.drawContours(img, [NumberPlateCnt], -1, (0, 255, 0), 3)

    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im)
    img3 = tk.Label(root, image=imgtk, height=x1, width=y1)
    img3.image = imgtk
    # img3.grid(column=2, row=1, sticky=tk.NE)
    img3.place(x=290, y=300)

def char_recg():
    # global fn
    # FName = fn
    # imgpath = FName

    cutx = 1
    cuty = 1

    Set2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
            'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    Dir = "36ClassSet"

    Net = NNtool(Dir, False)
    Net.SetSession()

    ip = sys.argv[0]

    # filename = "PLATES/plate-" + str(ip)

    fileName = askopenfilename(initialdir='/dataset', title='Select image',
                               filetypes=[("all files", "*.*")])
    #or_img = mpimg.imread('D:/gui_ppplate/cnn_code/plates/plate-13.jpg')  # Qriginal Image
    or_img = mpimg.imread(fileName)
    img = or_img[cutx:-cutx, cuty:-cuty]

    Per = (383 * 900 / img.size) ** (1 / 2)

    img = spm.imresize(img, (int(img.shape[0] * Per), int(img.shape[1] * Per)), 'bicubic')
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im)
    img1 = tk.Label(root, image=imgtk)
    img1.image = imgtk
    img1.grid(row=4, column=0, sticky=tk.NE)
    img1.place(x=400, y=110)

    if (len(img.shape) == 3):
        if (np.max(img) > 1):
            img = img
        img = img[:, :, 0] * 1 / 3 + img[:, :, 1] * 1 / 3 + img[:, :, 2] * 1 / 3

    # plt.title('rgb2gray')
    # plt.imshow(img, cmap='gray')
    # plt.axis('off')
    # plt.show()

    img = img.astype(int)

    HEinput = [70, 170, 170, 255, 5, 50]

    imgH, he_cr = HE(
        img,
        HEinput[0],
        HEinput[1],
        HEinput[2],
        HEinput[3],
        HEinput[4],
        HEinput[5]
    )

    if he_cr:
        # plt.subplot(2, 1, 1)
        # plt.title('Original Image')
        # plt.axis('off')
        # plt.imshow(img, cmap='Greys_r')
        # plt.subplot(2, 1, 2)
        plt.title('Histogram Equalization')
        # plt.axis('off')
        # plt.imshow(imgH, cmap='Greys_r')
    # plt.show()

    imgH = imgH / 255
    img = np.array(imgH)
    del imgH

    # Edge Operator
    Eimg, thita = Cany_skimage(img, 3, 0.1, 0.3)

    # plt.subplot(2, 1, 1)
    # # plt.axis('off')
    # # plt.imshow(Eimg, cmap='Greys_r')
    # # plt.title('Cany Opertor', fontsize=16)
    #
    # plt.subplot(2, 1, 2)
    # # plt.axis('off')
    # # plt.imshow(thita, cmap='Greys_r')
    # # plt.title('Angle Space', fontsize=16)
    # plt.show()

    Pon = GetPoints(Eimg)  # get points with non zero value

    ImPlace = np.zeros((img.shape[0], img.shape[1], 3))

    ImPlace[:, :, 0] = np.array(img)
    ImPlace[:, :, 1] = np.array(img)
    ImPlace[:, :, 2] = np.array(img)

    RectDB = DBs(Pon)  # DBSCAN

    RectHC = heightCluster(RectDB)  # Height Cluster

    Rect = AreaCluster(RectHC, img.shape)  # Area Cluster
    #
    # # plt.subplot(1, 3, 1)
    # # plt.axis('off')
    # # plt.title('DBSCAN')
    # RectsDrawing(ImPlace, RectDB, [1, 0, 0])
    #
    # # plt.subplot(1, 3, 2)
    # # plt.title('HeightCluster')
    # # plt.axis('off')
    # RectsDrawing(ImPlace, RectHC, [1, 0, 0])
    #
    # # plt.subplot(1, 3, 3)
    # # plt.title('AreaCluster')
    # # plt.axis('off')
    # RectsDrawing(ImPlace, Rect, [1, 0, 0])
    # # plt.subplots_adjust(0, 0, 1, 1)
    # # plt.show()
    del RectDB, RectHC

    #############
    scaleIM = 28
    Imd = 1  # image dimensions gray->1 rgb->3
    n_clusters_ = Rect.shape[0]
    gap = 2  # add black square around the image before scaling
    Net.Initialize_Vars()
    Label = []
    for I in range(0, n_clusters_):
        #############
        # CNN
        pos0x = Rect[I][0] - 1
        pos0y = Rect[I][1] - 1
        pos1x = Rect[I][2] + 1
        pos1y = Rect[I][3] + 1

        Label.append([pos0y])
        Rect1 = Eimg[pos0x:pos1x, pos0y:pos1y]
        thita1 = thita[pos0x:pos1x, pos0y:pos1y]
        #  fill edge binary images based on thita
        for i in range(Rect1.shape[0]):
            for j in range(Rect1.shape[1]):
                xstep = 0
                ystep = 0
                if thita1[i, j] != 0:
                    if thita1[i, j] == 1:
                        xstep += -1
                    elif thita1[i, j] == 2:
                        xstep += -1
                        ystep += -1
                    elif thita1[i, j] == 3:
                        ystep += -1
                    elif thita1[i, j] == 4:
                        xstep += 1
                        ystep += -1
                    elif thita1[i, j] == -1:
                        xstep += 1
                    elif thita1[i, j] == -2:
                        xstep += 1
                        ystep += 1
                    elif thita1[i, j] == -3:
                        ystep += 1
                    elif thita1[i, j] == -4:
                        xstep += -1
                        ystep += 1
                    ix = i + xstep
                    jy = j + ystep
                    if not (ix + xstep >= thita1.shape[0] or ix + xstep < 0 or jy + ystep < 0 or jy + ystep >=
                            thita1.shape[1]):
                        while thita1[ix + xstep, jy + ystep] == 0 and thita1[ix, jy + ystep] == 0 and thita1[
                            ix + xstep, jy] == 0:
                            if Rect1[ix, jy] == 1:
                                break
                            Rect1[ix, jy] = 1
                            ix += xstep
                            jy += ystep
                            if ix + xstep >= thita1.shape[0] or ix + xstep < 0 or jy + ystep < 0 or jy + ystep >= \
                                    thita1.shape[1]:
                                break
        # Rectangle image to square image adding black columns and rows, for better scaling
        dx = pos1x - pos0x
        dy = pos1y - pos0y
        a = abs(dx - dy) / 2
        z1 = int(a)
        z0 = int(a) + int(a != int(a))
        maxD = dx
        if (dx > dy):
            Rect1 = np.concatenate((np.zeros((dx, z0)), Rect1, np.zeros((dx, z1))), axis=1)
        elif (dx < dy):
            Rect1 = np.concatenate((np.zeros((z0, dy)), Rect1, np.zeros((z1, dy))), axis=0)
            maxD = dy
        Rect1 = np.concatenate((np.zeros((maxD, gap)), Rect1, np.zeros((maxD, gap))), axis=1)
        Rect1 = np.concatenate((np.zeros((gap, maxD + 2 * gap)), Rect1, np.zeros((gap, maxD + 2 * gap))), axis=0)

        ###########

        ScalePic = ScaleArray(Rect1, scaleIM - 2 * gap)
        ScalePic = np.concatenate((np.zeros((scaleIM - 2 * gap, gap)), ScalePic, np.zeros((scaleIM - 2 * gap, gap))),
                                  axis=1)
        ScalePic = np.concatenate((np.zeros((gap, scaleIM)), ScalePic, np.zeros((gap, scaleIM))), axis=0)

        Score = Net.Layers[-1].eval(feed_dict={
            Net.Layers[0]: ScalePic.reshape(1, scaleIM, scaleIM, Imd),
            Net.keep_prob: np.ones((Net.DroupoutsProbabilitys.shape[0]))
        })

        Conf = np.max(Score)
        plate_char = Set2[np.where(Score.reshape(36) == Conf)[0][0]]
        Label[-1].append(plate_char)

        # plt.subplot(int(np.ceil(n_clusters_/5)), min(n_clusters_, 5), I+1)
        # plt.title("Confidence:\n %.2f\nClass: %s" % (Conf, plate_char))
        # plt.axis('off')
        # plt.imshow(ScalePic, cmap='Greys_r')

    # plt.subplots_adjust(wspace=1)
    # plt.show()

    plate_number = ''
    for cr in sorted(Label, key=lambda Labeli: Labeli[0]):
        plate_number += cr[1]

    # display number plate digits
    #plt.title(plate_number)
    print(plate_number)
    # plt.imshow(or_img)
    # plt.axis('off')
    # plt.show()

    import time
    start = time.time()
    a = range(100000)
    b = []
    for i in a:
        b.append(i * 2)
    end = time.time()
    print(end - start)

    # import sys
    # import subprocess
    # global fn
    # FName = fn
    # imgpath = FName
    # image_out = subprocess.check_output([sys.executable, "single_plate.py", "34"])
    # print(image_out)
    # label_output = tk.Label(root, text='Recogning...Please Wait')  # , height="20", width="10")
    # label_output.grid(column=0, row=2, padx=10, pady=10)
    # label_output.place(x=500, y=300)
    #
    # label_output.config(text= image_out)
    #from subprocess import call
    #call(["python", "single_plate.py"])


input_image = tk.Button(frame_alpr, text="RECTANGLE", command=img_input,width=20, height=1, font=('times', 15, ' bold '))
input_image.place(x=10, y=40)

preprocessing = tk.Button(frame_alpr, text="CIRCLE", command=preprocessing, width=20, height=1, font=('times', 15, ' bold '))
preprocessing.place(x=10, y=100)

edge_detection = tk.Button(frame_alpr, text="STAR", command=edges, width=20, height=1, font=('times', 15, ' bold '))
edge_detection.place(x=10, y=160)

license_plate_loc = tk.Button(frame_alpr, text="License Plate Localization", command=lic_plate_loc,width=20, height=1, font=('times', 15, ' bold '))
license_plate_loc.place(x=10, y=220)

char_recog = tk.Button(frame_alpr, text="Character Recognition", command=char_recg,width=20, height=1, font=('times', 15, ' bold '))
char_recog.place(x=10, y=280)

ip_img = tk.Button(root, text="Input Vehicle Image", command=char_recg,width=20, height=1, font=('times', 15, ' bold '))
ip_img.place(x=500, y=60)

exit = tk.Button(frame_alpr, text="Exit", command=root.destroy, width=20, height=1, font=('times', 15, ' bold '))
exit.place(x=10, y=340)

train_database = tk.Button(root, text="Train Database", command=img_input,width=15, height=1, font=('times', 15, ' bold '))
train_database.place(x=1050, y=60)








root.mainloop()