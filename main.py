__author__ = 'SQUAD'

import numpy as np
import cv2
from matplotlib import pyplot as plt

def match(coin, value):

    change = 0.00
    tempImg = cv2.imread('input.jpg', 0)
    w, h = coin.shape[::-1]
    cv2.imwrite('blah.jpg', tempImg)

    for x in range(0,100):
        tempImg = cv2.imread('blah.jpg', 0)
        res = cv2.matchTemplate(tempImg,coin,cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # print min_val
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # Draw rectangle on image where the best score is found

        cv2.rectangle(tempImg,top_left, bottom_right, 255, -5)

        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(tempImg,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

        if x > 0:
            if value == 0.01 and max_val < 10000000:
                return change
            elif value == .05 and max_val < 50000000:
                return change
            elif value == 0.10 and max_val < 20000000:
                return change
            elif value == 0.25 and max_val < 80000000:
                return change

        cv2.imwrite('blah.jpg', tempImg)
        change += value

def getPennies():

    penny = cv2.imread('penny.jpg', 0)
    return match(penny, 0.01)

def getNickels():

    nickel = cv2.imread('nickel.jpg', 0)
    return match(nickel, 0.05)

def getDimes():

    dime = cv2.imread('dime.jpg', 0)
    return match(dime, 0.10)

def getQuarters():

    quarter = cv2.imread('quarter.jpg', 0)
    return match(quarter, 0.25)
def main():
   print getPennies() + getNickels() + getDimes() + getQuarters()

# Code runs here
main()