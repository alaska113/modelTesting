import numpy as np
import matplotlib.pyplot as plt
from AppKit import NSScreen
from matplotlib.widgets import Button
import cv2
import os


class MatrixTest:
    def __init__(self):
        self.width = NSScreen.mainScreen().frame().size.width
        self.height = NSScreen.mainScreen().frame().size.height
        self.points = self.createTestPoints()

    def createTestPoints(self):
        testPoints = [
                        (int(self.width/6), int(self.height/6)),
                        (int(self.width/2), int(self.height/6)),
                        (int(self.width-(self.width/6)), int(self.height/6)),
                        (int(self.width-(self.width/6)), int(self.height/2)),
                        (int(self.width/2), int(self.height/2)),
                        (int(self.width/6), int(self.height/2)),
                        (int(self.width/6), int(self.height-(self.height/6))),
                        (int(self.width/2), int(self.height-(self.height/6))),
                        (int(self.width-(self.width/6)), int(self.height-(self.height/6)))

        ]
        return testPoints
