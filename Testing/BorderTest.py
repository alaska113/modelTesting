import numpy as np
import matplotlib.pyplot as plt
from AppKit import NSScreen
from matplotlib.widgets import Button
import cv2
import os


class BorderTest:
    def __init__(self):
        self.width = NSScreen.mainScreen().frame().size.width
        self.height = NSScreen.mainScreen().frame().size.height
        self.points = self.createTestPoints()

    def createTestPoints(self):
        testPoints = [
                        (0,0),
                        (int(self.width/2), 0),
                        (int(self.width), 0),
                        (int(self.width), int(self.height/2)),
                        (int(self.width), int(self.height)),
                        (int(self.width/2), int(self.height)),
                        (0, int(self.height)),
                        (0, int(self.height/2))

        ]
        return testPoints
