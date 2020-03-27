from Testing.RunTesting import RunTesting
from Testing.MatrixTest import MatrixTest
from Testing.BorderTest import BorderTest
from preProcessing.ProcessImages import ProcessImages
from ModelModules.GazePredictor import GazePredictor
import sys


def runTests():
    if sys.argv[1] == "Matrix":
        test = MatrixTest()

    elif sys.argv[1] == "Border":
        test = BorderTest()
    else:
        print("Not a valid test")
        return 0

    RunTesting(test)
    ProcessImages("./imgs/*.jpg", ".")
    predictor = GazePredictor()
    predictor.runOnCompressedData("./demoData.npz")




if __name__ =="__main__":
    runTests()
