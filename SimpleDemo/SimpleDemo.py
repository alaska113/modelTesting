import glob
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter

class SimpleDemo:
    def __init__(self, demoDataPath):
        self.demoDataPath = demoDataPath
        # self.imagesDir = self.demoDataPath + "/frames/"
        self.getDemoData()
        self.plotGazeData()


    def plotGazeData(self):
        metadata = dict(title="Moive Test", artist="khan", comment="test")
        writer = FFMpegWriter(fps=15, metadata=metadata)
        fig = plt.figure()
        ax = plt.gca()

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])

        with writer.saving(fig, "demoTest.mp4", len(self.eyes)/2.0):
            for i in range(0, len(self.eyes), 2):
                x = self.gazes[i][0]
                y = self.gazes[i][1]
                plt.plot(x, y, 'r.')
                writer.grab_frame()

    def getDemoData(self):
        # self.images = self.loadImages()
        allData = self.loadData()
        self.eyes = allData["eyes"]
        self.poses = allData["poses"]
        self.gazes = allData["gazes"]

    def loadImages(self):
        filenames = glob.glob(self.imagesDir)
        filenames.sort()
        images = [cv2.imread(img) for img in filenames]
        return images

    def loadData(self):
        loaded = np.load(self.demoDataPath+"outputs.npz")
        return loaded


if __name__ == "__main__":
    SimpleDemo("./")
