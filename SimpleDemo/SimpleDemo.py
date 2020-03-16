import glob
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
import matplotlib.image as mpimg

class SimpleDemo:
    def __init__(self, demoDataPath):
        self.demoDataPath = demoDataPath
        self.imagesDir = self.demoDataPath + "frames/"
        self.getDemoData()
        self.plotGazeData()


    def plotGazeData(self):
        metadata = dict(title="Moive Test", artist="khan", comment="test")
        writer = FFMpegWriter(fps=30, metadata=metadata)
        fig = plt.figure(constrained_layout=True)
        # ax = plt.gca()
        gs = fig.add_gridspec(3,3)

        #
        # ax.set_xlim([-1, 1])
        # ax.set_ylim([-1, 1])

        with writer.saving(fig, "demoTest.mp4", len(self.eyes)/2.0):
            # for i in range(0, len(self.eyes), 2):
            for i in range(0, len(self.images)):
                if i == 0:
                    j=i
                else:
                    j = i+1

                print("i: ", i)
                print("j: ", j)
                print("----")
                x = self.gazes[j][0]
                y = self.gazes[j][1]

                f_ax1 = fig.add_subplot(gs[0,1])
                # ax1 = plt.subplot(121)
                f_ax1.imshow(self.images[i])

                f_ax2 = fig.add_subplot(gs[1:, :])
                # ax2 = plt.subplot(122)
                f_ax2.set_xlim(-1, 1)
                f_ax2.set_ylim(-1, 1)
                f_ax2.plot(x, y, 'r.')
                writer.grab_frame()

    def getDemoData(self):
        self.images = self.loadImages()
        allData = self.loadData()
        self.eyes = allData["eyes"]
        self.poses = allData["poses"]
        self.gazes = allData["gazes"]

    def loadImages(self):
        images = []
        filenames = glob.glob(self.imagesDir + "*.jpg")
        print("Checking what we're grabbing: ", filenames[0][13:-4])
        filenames = sorted(filenames, key=lambda s: int(s[13:-4]))
        # filenames.sort()
        print(filenames)
        for img in filenames:
            image = mpimg.imread(img)
            # image = np.array(image, dtype='float32')
            # # image.float()
            images.append(image)
        # images = [cv2.imread(img) for img in filenames]
        return images

    def loadData(self):
        loaded = np.load(self.demoDataPath+"outputs.npz")
        return loaded


if __name__ == "__main__":
    SimpleDemo("./demo/")
