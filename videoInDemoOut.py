from preProcessing.FrameGrabber import FrameGrabber
from preProcessing.ProcessImages import ProcessImages
from ModelModules.GazePredictor import GazePredictor
# from ModelModules.MPIIGazeDataset import MPIIGazeDataset
from SimpleDemo.SimpleDemo import SimpleDemo
import sys

class VideoInDemoOut:
    def __init__(self, videoPath):
        self.videoPath = videoPath
        self.frameGrabber = FrameGrabber(self.videoPath)
        self.imageProcessor = ProcessImages("./videoData/*.jpg")
        self.gazePredictor = GazePredictor(self.imageProcessor.eyes, self.imageProcessor.poses)
        self.simpleDemo = SimpleDemo("./")

if __name__ == "__main__":
    VideoInDemoOut(sys.argv[1])
