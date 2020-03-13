from preProcessing.FrameGrabber import FrameGrabber
from preProcessing.ProcessImages import ProcessImages
from ModelModules.GazePredictor import GazePredictor
# from ModelModules.MPIIGazeDataset import MPIIGazeDataset
from SimpleDemo.SimpleDemo import SimpleDemo
import sys
import os

class VideoInDemoOut:
    def __init__(self, videoPath):
        self.videoPath = videoPath
        self.makeDemoDir()
        self.frameGrabber = FrameGrabber(self.videoPath, "./demo")
        self.imageProcessor = ProcessImages("./demo/frames/*.jpg", "./demo")
        self.gazePredictor = GazePredictor(self.imageProcessor.eyes, self.imageProcessor.poses)
        self.simpleDemo = SimpleDemo("./demo/")

    def makeDemoDir(self):
        try:
            if not os.path.exists("./demo"):
                os.makedirs("./demo")

        except OSError:
            print("Error creating directory for frames from video!")

if __name__ == "__main__":
    VideoInDemoOut(sys.argv[1])
