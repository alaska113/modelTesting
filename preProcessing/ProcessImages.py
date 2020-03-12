
import importlib
import dlib
import cv2
from imutils import face_utils
import matplotlib.pyplot as plt
import numpy as np
import math
import glob
from statistics import mean
import sys

class ProcessImages:
    def __init__(self, imagesDir):
        self.imagesDir = imagesDir
        self.images = self.loadImages()
        self.eyes = []
        self.poses = []
        self.processImages()
        self.saveData()

    def loadImages(self):
        filenames = glob.glob(self.imagesDir)
        filenames.sort()
        images = [cv2.imread(img) for img in filenames]
        return images

    def saveData(self):
        np.savez_compressed("./demoData", eyes=self.eyes, poses=self.poses)

    def loadData(self):
        loaded = np.load("./demoData.npz")

    def processImages(self):
        counter = 0
        for image in self.images:
            counter+=1
            faceFound, shapes, grayImage = self.faceDetection(image)
            if faceFound:
                pose, leftEye, rightEye = self.extractFeaturesFromImage(shapes, grayImage)
                self.poses.append(pose)
                self.poses.append(pose) #Have to do twice, for both eyes.
                self.eyes.append(leftEye)
                self.eyes.append(rightEye)
        self.eyes = np.array(self.eyes)
        self.poses = np.array(self.poses)



    def faceDetection(self, image):
        shapes, grayImage = self.getFacialLandmarks(image)
        if len(shapes):
            return True, shapes, grayImage
        else:
            return False, shapes, grayImage



    def extractFeaturesFromImage(self, shapes, grayImage):
        landmarks = shapes[0] #TODO this assumes there is only one face in image!!
        leftEye, rightEye = self.extractEyesFromGrayscale(grayImage, landmarks)
        #TODO Need to change hardcoded resolution for YPR
        yaw, pitch, roll = self.getHeadPosition(shapes[0], (426,640))
        return [yaw, roll], leftEye, rightEye

    def convertImageToGrayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def getFacialLandmarks(self, image):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("./preProcessing/shape_predictor_68_face_landmarks.dat")
        grayImage = self.convertImageToGrayscale(image)
        rects = detector(grayImage)
        shapes = []
        for rect in rects:
            shape = predictor(grayImage,rect)
            shape = face_utils.shape_to_np(shape)
            shapes.append(shape)
        return shapes, grayImage

    def extractEyesFromGrayscale(self, grayImage, landmarks):
        leftEye = self.getLeftEye(grayImage, landmarks)
        rightEye = self.getRightEye(grayImage, landmarks)
        return leftEye, rightEye

    def getRightEye(self, grayImage, landmarks):
        desiredSize = [60, 36]
        middlePoint = (mean([landmarks[39][0], landmarks[36][0]]), mean([landmarks[39][1], landmarks[36][1]]))
        upLeft = (int(middlePoint[0] - desiredSize[0]/2), int(middlePoint[1] + desiredSize[1]/2))
        botRight = (int(middlePoint[0] + desiredSize[0]/2), int(middlePoint[1] - desiredSize[1]/2))
        croppedEyeGray = grayImage[botRight[1]:upLeft[1], upLeft[0]:botRight[0]]
        croppedEyeGray = np.array(croppedEyeGray, dtype=np.float32).flatten()
        croppedEyeGray = self.normalize(croppedEyeGray)
        croppedEyeGray = np.resize(np.array(croppedEyeGray), (36,60))
        return croppedEyeGray

    def getLeftEye(self, grayImage, landmarks):
        desiredSize = [60, 36]
        middlePoint = (mean([landmarks[42][0], landmarks[45][0]]), mean([landmarks[42][1], landmarks[45][1]]))
        upLeft = (int(middlePoint[0] - desiredSize[0]/2), int(middlePoint[1] + desiredSize[1]/2))
        botRight = (int(middlePoint[0] + desiredSize[0]/2), int(middlePoint[1] - desiredSize[1]/2))
        croppedEyeGray = grayImage[botRight[1]:upLeft[1], upLeft[0]:botRight[0]]
        croppedEyeGray = np.array(croppedEyeGray, dtype=np.float32).flatten()
        croppedEyeGray = self.normalize(croppedEyeGray)
        croppedEyeGray = np.resize(np.array(croppedEyeGray), (36,60))
        return croppedEyeGray


    def getHeadPosition(self, landmarksArr, resolution):
        landmarks = np.array(
            [
                (landmarksArr[45][0], landmarksArr[45][1]),
                (landmarksArr[36][0], landmarksArr[36][1]),
                (landmarksArr[33][0], landmarksArr[33][1]),
                (landmarksArr[54][0], landmarksArr[54][1]),
                (landmarksArr[48][0], landmarksArr[48][1]),
                (landmarksArr[8][0],  landmarksArr[8][1]),
            ], dtype=np.float,
        )
        image_points = np.array(
            [
                (landmarks[2][0], landmarks[2][1]),
                (landmarks[5][0], landmarks[5][1]),
                (landmarks[0][0], landmarks[0][1]),
                (landmarks[1][0], landmarks[1][1]),
                (landmarks[3][0], landmarks[3][1]),
                (landmarks[4][0], landmarks[4][1]),
            ], dtype=np.float,
        )
        for point in image_points:
            point = self.rotate_landmark(image_points[0], point, np.pi)

        model_points = np.array(
            [
                (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-165.0, 170.0, -135.0),
                (165.0, 170.0, -135.0), (
                    -150.0, -
                    150.0, -125.0,
                ), (150.0, -150.0, -125.0),
            ], dtype=np.float,
        )
        # Camera internals
        center = (resolution[1]/2, resolution[0]/2)
        focal_length = center[0] / np.tan((60.0/2.0) * (np.pi / 180.0))
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ], dtype=np.float,
        )
        # Assuming no lens distortion
        dist_coeffs = np.zeros((4, 1), dtype="double",)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE,
        )

        axis = np.float32([
            [500, 0, 0],
            [0, 500, 0],
            [0, 0, 500],
        ])

        imgpts, jac = cv2.projectPoints(
            axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs,
        )
        modelpts, jac2 = cv2.projectPoints(
            model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs,
        )
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))

        return yaw, pitch, roll


    def rotate_landmark(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

    def normalize(self, grayscale):
        return (grayscale - min(grayscale))/(max(grayscale)-min(grayscale))


if __name__ == "__main__":
    PreProcessImages(sys.argv[1])
