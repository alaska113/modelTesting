from ModelModules.preProcessing.FrameGrabber import FrameGrabber
from ModelModules.preProcessing.ProcessImages import ProcessImages
from ModelModules.GazePredictor import GazePredictor
from ModelModules.MPIIGazeDataset import MPIIGazeDataset
from SimpleDemo.SimpleDemo import SimpleDemo

class VideoInDemoOut(self):
    def __init__(self, videoPath):
        self.videoPath = videoPath
        self.frameGrabber = FrameGrabber(self.videoPath)
        self.imageProcessor = ProcessImages("./videoData/*.jpg")
        self.gazePredictor = GazePredictor(self.imageProcessor.eyes, self.imageProcessor.poses)
        self.simpleDemo = SimpleDemo("./ModelModules/outputs.npz")
