import numpy as np
import matplotlib.pyplot as plt
from AppKit import NSScreen
from matplotlib.widgets import Button
import cv2
import os


class RunTesting:
    def __init__(self, test):
        self.test = test
        self.width = NSScreen.mainScreen().frame().size.width
        self.height = NSScreen.mainScreen().frame().size.height
        self.captureDevice = cv2.VideoCapture(0)
        self.testImage = self.createTestImage()
        self.makeImgsDir()
        self.recordVideo()


    def recordVideo(self):
        frameCounter = 0
        pointCounter = 0
        specialFrames = []

        while True:
            cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.imshow("window", self.testImage)
            cv2.waitKey(0)
            ret, frame = self.captureDevice.read()
            if ord("a"):
                if ret:
                    cv2.imwrite("./imgs/img" + str(frameCounter) + ".jpg", frame)
                if pointCounter > len(self.test.points)-1:
                    break
                self.updatePoints(pointCounter)
                specialFrames.append(frameCounter)
                pointCounter+=1

            frameCounter+=1

        cv2.destroyAllWindows()


    # def loopThroughPoints(self):
    #     c = 0
    #     while True:
    #         cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    #         cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    #         # self.firstTestImageStart = cv2.circle(self.firstTestImageStart, self.testPoints[0], 5, (0,0,255), -1)
    #         cv2.imshow("window", self.firstTestImageStart)
    #         cv2.waitKey(0)
    #         if c <= len(self.testPoints)-1:
    #             # self.firstTestImageStart = cv2.circle(self.firstTestImageStart, self.testPoints[c], 5, (0,0,255), -1)
    #             self.updatePoints(c)
    #             if ord("a"):
    #                 ret, frame = self.captureDevice.read()
    #                 if ret == True:
    #                     cv2.imwrite("./imgs/img" + str(c) + ".jpg", frame)
    #                 c+=1
    #         else:
    #             break
    #     cv2.destroyAllWindows()

    def updatePoints(self,i):
        for point in self.test.points:
            self.testImage = cv2.circle(self.testImage, point, 8, (0,0,0), -1)
        self.testImage = cv2.circle(self.testImage, self.test.points[i], 8, (0,0,255), -1)


    def createTestImage(self):
        self.createBlankScreen()
        img = self.createTestScreen()
        return img

    def createBlankScreen(self):
        arr = [[(255,255,255)] * int(self.width)] * int(self.height)
        img = np.array(arr)
        cv2.imwrite("window.jpg", img)


    def createTestScreen(self):
        img = cv2.imread('window.jpg')
        c = 0
        for point in self.test.points:
            img = cv2.circle(img, point, 8, (0,0,0), -1)
        return img

    def makeImgsDir(self):
        try:
            if not os.path.exists("./imgs"):
                os.makedirs("./imgs")

        except OSError:
            print("Error creating imgs directory")


if __name__ == "__main__":
    RunTesting()
