__author__ = 'AlexisT'

import numpy as np
import cv2
from cv2 import cv
from matplotlib import pyplot as plt


def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def videoInput():
    global out
    vid = cv2.VideoCapture(0)

    while True:
        ret, frame = vid.read()
        cv2.imshow('video test', frame)
        if ret:    # frame captured without any errors
            out = cv2.imwrite('out.jpg',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()

    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 720)
    #
    # while(True):
    #     ret, frame = cap.read()
    #     out = cv2.imwrite('out.jpg',frame)
    #     roi = frame[0:500, 0:500]
    #     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #
    #     gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
    #     thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #         cv2.THRESH_BINARY_INV, 11, 1)
    #
    #     kernel = np.ones((3, 3), np.uint8)
    #     closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
    #         kernel, iterations=4)
    #
    #     cont_img = closing.copy()
    #     contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL,
    #         cv2.CHAIN_APPROX_SIMPLE)
    #
    #     for cnt in contours:
    #         area = cv2.contourArea(cnt)
    #         if area < 2000 or area > 4000:
    #             continue
    #
    #         if len(cnt) < 5:
    #             continue
    #
    #         ellipse = cv2.fitEllipse(cnt)
    #         cv2.ellipse(roi, ellipse, (0,255,0), 2)
    #
    #     cv2.imshow("Morphological Closing", closing)
    #     cv2.imshow("Adaptive Thresholding", thresh)
    #     cv2.imshow('Contours', roi)
    #
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # cap.release()
    # cv2.destroyAllWindows()

def match():
    global change
    total = 0
    value = 0.00
    change = 0.00
    camImg2 = cv2.imread("out.jpg", 0)
    # camImg2 = cv2.imread("out.jpg", 1)

    MIN_MATCH_COUNT = 2
    goodPen = 0
    goodNic = 0
    goodDim = 0
    goodQtr = 0

    for db in database:
        print db
        total = total + database[db]
        avgScore = total/26

        coin2 = cv2.imread(db + ".jpg", 0)
        w, h = coin2.shape[::-1]
        camImg2 = cv2.resize(camImg2,(w,h,))

        cv2.imwrite('camImg2.jpg', camImg2)

        coin = cv2.cvtColor(coin2,cv2.COLOR_GRAY2RGB)
        camImg = cv2.cvtColor(camImg2,cv2.COLOR_GRAY2RGB)
        cv2.imwrite('camImg.jpg', camImg)


        # Template Matching
        res = cv2.matchTemplate(camImg,coin,cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        norm = 1 - min_val

        threshold = 0.8
        loc = np.where( res >= threshold)
        norm = float(res[0])
        for pt in zip(*loc[::-1]):
            cv2.rectangle(camImg, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

        # circles = cv2.HoughCircles(camImg2, cv2.cv.CV_HOUGH_GRADIENT, 2, 10, np.array([]), 20, 60, w/10)[0]
        # red = (0,0,255)
        # r = 45
        # x, y = np.meshgrid(np.arange(w), np.arange(h))
        # for c in circles[0:]:
        #     cv2.circle(camImg, (c[0],c[1]), c[2], red, 2)

            
        # top_left = min_loc
        # bottom_right = (top_left[0] + w, top_left[1] + h)
        # # Draw rectangle on image where the best score is found
        # cv2.rectangle(camImg, top_left, bottom_right, 255, 2)
        #  # Display image using OpenCV
        # cv2.imshow(db+'Match Found',camImg)
        # Wait for user to close window
        # cv2.waitKey()


        # Histogram
        hist1 = cv2.calcHist([camImg],[0],None,[256],[0,256])
        hist2 = cv2.calcHist([coin],[0],None,[256],[0,256])
        hist3 = cv2.compareHist(hist1, hist2, cv2.cv.CV_COMP_CORREL)

        # Size Check
        camImg = cv2.cvtColor(camImg,cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(camImg,(5,5),0)
        thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        print len(contours)

        cv2.drawContours(camImg,contours,-1,(255,0,0),2)

        conImg = camImg
        cv2.imwrite('contours.jpg', conImg)


        if(norm > .60) and (database[db] == 1) and (len(contours) > 25 and len(contours) <= 50 ):# and (len(good)>=MIN_MATCH_COUNT): #and (hist3>= .75) and (avgScore>=0 and avgScore <= 1):
            goodPen += 1

        if(norm > .60)  and (database[db] == 2) and (len(contours) > 50 and len(contours) <= 75 ):# and (len(good)>=MIN_MATCH_COUNT):# and (hist3>= .75):# and (avgScore>1 and avgScore <= 2):
            goodNic += 1

        if(norm > .60) and (database[db] == 3) and (len(contours) >= 0 and len(contours) <= 25 ):# and (len(good)>=MIN_MATCH_COUNT): # and (avgScore <= 3):
            goodDim += 1

        if(norm > .60) and (database[db] == 4) and (len(contours) > 75 and len(contours) <= 750 ):# and (len(good)>=MIN_MATCH_COUNT):# and (avgScore <= 4):
            goodQtr += 1



        print('database[db]',database[db])
        print ('change:',change)
        print ('value:',value)
        print('norm:', norm)
        # print('len good:',len(good))
        print('hist3:',hist3)
        print('avgScore:',avgScore)
        print('goodPen', goodPen)
        print('goodNic', goodNic)
        print('goodDim', goodDim)
        print('goodQtr', goodQtr)

        cv2.imwrite('camImg.jpg', camImg)
        # change += value
    if (goodPen >= 10):
        value = 0.01
        change += value
    if(goodNic >= 10):
        value = 0.05
        change += value
    if (goodDim >= 10):
        value = 0.10
        change += value
    if(goodQtr >= 10):
        value = 0.25
        change += value
    cv2.imshow("Original Camera Image", out)
    cv2.imshow("Contours", conImg)
    cv2.waitKey()

    return change

def main():
    global database
    global db
    global dbval
    database = {'penny (1)': 1, 'penny (2)': 1, 'penny (3)': 1, 'penny (4)': 1, 'penny (5)': 1,
                'penny (6)': 1, 'penny (7)': 1, 'penny (8)': 1, 'penny (9)': 1, 'penny (10)': 1,
                'penny (11)': 1, 'penny (12)': 1, 'penny (13)': 1, 'penny (14)': 1, 'penny (15)': 1,
                'penny (16)': 1, 'penny (17)': 1, 'penny (18)': 1, 'penny (19)': 1, 'penny (20)': 1,
                'penny (21)': 1, 'penny (22)': 1, 'penny (23)': 1, 'penny (24)': 1, 'penny (25)': 1,
                'penny (26)': 1,
                'nickel (1)': 2, 'nickel (2)': 2, 'nickel (3)': 2, 'nickel (4)': 2, 'nickel (5)': 2,
                'nickel (6)': 2, 'nickel (7)': 2, 'nickel (8)': 2, 'nickel (9)': 2, 'nickel (10)': 2,
                'nickel (11)': 2, 'nickel (12)': 2, 'nickel (13)': 2, 'nickel (14)': 2, 'nickel (15)': 2,
                'nickel (16)': 2, 'nickel (17)': 2, 'nickel (18)': 2, 'nickel (19)': 2, 'nickel (20)': 2,
                'nickel (21)': 2, 'nickel (22)': 2, 'nickel (23)': 2, 'nickel (24)': 2, 'nickel (25)': 2,
                'nickel (26)': 2,
                'dime (1)': 3, 'dime (2)': 3, 'dime (3)': 3, 'dime (4)': 3, 'dime (5)': 3,
                'dime (6)': 3, 'dime (7)': 3, 'dime (8)': 3, 'dime (9)': 3, 'dime (10)': 3,
                'dime (11)': 3, 'dime (12)': 3, 'dime (13)': 3, 'dime (14)': 3, 'dime (15)': 3,
                'dime (16)': 3, 'dime (17)': 3, 'dime (18)': 3, 'dime (19)': 3, 'dime (20)': 3,
                'dime (21)': 3, 'dime (22)': 3, 'dime (23)': 3, 'dime (24)': 3, 'dime (25)': 3,
                'dime (26)': 3,
                'quarter (1)': 4, 'quarter (2)': 4, 'quarter (3)': 4, 'quarter (4)': 4, 'quarter (5)': 4,
                'quarter (6)': 4, 'quarter (7)': 4, 'quarter (8)': 4, 'quarter (9)': 4, 'quarter (10)': 4,
                'quarter (11)': 4, 'quarter (12)': 4, 'quarter (13)': 4, 'quarter (14)': 4, 'quarter (15)': 4,
                'quarter (16)': 4, 'quarter (17)': 4, 'quarter (18)': 4, 'quarter (19)': 4, 'quarter (20)': 4,
                'quarter (21)': 4, 'quarter (22)': 4, 'quarter (23)': 4, 'quarter (24)': 4, 'quarter (25)': 4,
                'quarter (26)': 4,}
    db = database.items()
    videoInput()
    match()

    print 'Total Change: {}'.format(change)


# Code runs here
main()