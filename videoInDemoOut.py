from ModelModules.preProcessing.main import *
from ModelModules.GazePredictor import GazePredictor
from ModelModules.MPIIGazeDataset import MPIIGazeDataset

class VideoInDemoOut(self):
    def __init__(self, videoPath):
        self.videoPath = videoPath
        
