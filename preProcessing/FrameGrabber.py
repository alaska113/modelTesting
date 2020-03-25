import cv2
import os
import sys

class FrameGrabber:
    def __init__(self, videoPath, savePath):
        self.videoPath = videoPath
        self.savePath = savePath + "/frames"
        self.video = cv2.VideoCapture(videoPath)
        self.writeFrameDirectory()
        self.grabFrames()

    def writeFrameDirectory(self):
        try:
            if not os.path.exists(self.savePath):
                os.makedirs(self.savePath)

        except OSError:
            print("Error creating directory for frames from video!")

    def grabFrames(self):
        currentFrame = 0

        while(True):
            ret, frame = self.video.read()

            if ret:
                name = self.savePath + '/' + str(currentFrame) + '.jpg'
                print("Creating..." + name)
                cv2.imwrite(name, frame)
                currentFrame += 1

            else:
                break

        self.video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    videoPath = sys.argv[1]
    savePath = sys.argv[2]
    FrameGrabber(videoPath, savePath)
