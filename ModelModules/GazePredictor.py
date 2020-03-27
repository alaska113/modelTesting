import importlib
import torch
import torch.utils.data
import numpy as np
import sys
from ModelModules.MPIIGazeDataset import MPIIGazeDataset

class GazePredictor:
    def __init__(self):
        self.dummy = None

    def run(self):
        self.model = self.loadModel()
        # self.model.cuda()
        test_loader = self.get_loader(self.eyes, self.poses, len(self.eyes), 1, True)
        self.outputs = self.test(test_loader)

    def runOnCompressedData(self, path):
        self.dataFilePath = path
        self.data = np.load(self.dataFilePath)
        self.model = self.loadSavedModel()
        test_loader = self.get_loader(
                        self.data["eyes"],
                        self.data["poses"],
                        len(self.data["eyes"]), 1, True)

        self.outputs = self.test(test_loader)
        print(self.outputs)

    def saveOutputs(self, outputs, images, poses):
        np.savez_compressed("./demo/outputs", gazes=outputs, eyes=images, poses=poses)

    def loadSavedModel(self):
        module = importlib.import_module('models.{}'.format("lenet"))
        model = module.Model()
        # model.load_state_dict(torch.load("./results/00/config.json", map_location=torch.device('cpu')))
        modelLoaded = torch.load("./results/00/model_state.pth", map_location=torch.device('cpu'))["state_dict"]
        model.load_state_dict(modelLoaded)
        return model

    def loadModel(self):
        module = importlib.import_module('models.{}'.format("lenet"))
        model = module.Model()
        return model

    def test(self, test_loader):
        self.model.eval()

        for step, (images, poses) in enumerate(test_loader):
            images = images.float()
            poses = poses.float()

            with torch.no_grad():
                outputs = self.model(images, poses)

        self.saveOutputs(outputs, images, poses)
        return outputs


    def get_loader(self, images, poses, batch_size, num_workers, use_gpu):
        test_dataset = MPIIGazeDataset(images, poses)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=use_gpu,
            drop_last=False,
        )
        return test_loader





if __name__ == "__main__":
    GazePredictor("/Users/khan/code/attently/modelTesting/demoData.npz")
