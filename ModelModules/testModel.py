import importlib
import torch
import torch.utils.data
import numpy as np
import sys

class GazePredictor:
    def __init__(self, dataFilePath):
        self.dataFilePath = dataFilePath

    def runOnCompressedData(self):
        self.data = np.load(self.dataFilePath)
        self.model = self.loadModel()
        self.model.cuda()
        test_loader = self.get_loader(
                        self.data["eyes"],
                        self.data["poses"],
                        len(self.data["eyes"]), 1, True)

        self.outputs = self.test(test_loader)

    def saveOutputs(self, outputs, images, poses):
        np.savez_compressed("./outputs", gazes=outputs, eyes=images, poses=poses)

    def loadModel(self):
        module = importlib.import_module('models.{}'.format("lenet"))
        model = module.Model()
        return model

    def test(self, test_loader):
        self.model.eval()

        for step, (images, poses) in enumerate(test_loader):
            images = images.float()
            poses = poses.float()
            images = images.cuda()
            poses = poses.cuda()

            with torch.no_grad():
                outputs = self.model(images, poses)

        self.saveOutputs(outputs.cpu(), images.cpu(), poses.cpu())
        return outputs.cpu()


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



class MPIIGazeDataset(torch.utils.data.Dataset):
    def __init__(self, images, poses):
        self.images = images
        self.poses = poses
        self.length = len(self.images)
        self.images = torch.unsqueeze(torch.from_numpy(self.images), 1)
        self.poses = torch.from_numpy(self.poses)

    def __getitem__(self, index):
        return self.images[index], self.poses[index]

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__





if __name__ == "__main__":
    GazePredictor("/home/khan/modelTesting/ModelModules/preProcessing/demoData.npz")
