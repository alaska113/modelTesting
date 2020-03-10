import importlib
import torch
import torch.utils.data
import numpy as np
import sys

def runOnCompressedData(dataFilePath):
    loaded = np.load(dataFilePath)
    model = loadModel()
    model.cuda()
    test_loader = get_loader(loaded["eyes"], loaded["poses"], len(loaded["eyes"]), 1, True)
    outputs = test(model, test_loader)
    

def saveOutputs(outputs, images, poses):
    np.savez_compressed("./outputs", gazes=outputs, eyes=images, poses=poses) 
    
def run():
    imagePath = sys.argv[1]
    data = PreProcessImages(imagePath)
    model = loadModel()
    model.cuda()
    test_loader = get_loader(data.eyes, data.poses, len(data.eyes), 1, True)
    test(model, test_loader)


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

def get_loader(images, poses, batch_size, num_workers, use_gpu):  
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



def loadModel():
    # model
    module = importlib.import_module('models.{}'.format("lenet"))
    model = module.Model()
    return model

def test(model, test_loader):
    model.eval()
    
    for step, (images, poses) in enumerate(test_loader):
        print("Step: ", step)
        images = images.float()
        poses = poses.float()
        images = images.cuda()
        poses = poses.cuda()

        with torch.no_grad():
            outputs = model(images, poses)

    saveOutputs(outputs.cpu(), images.cpu(), poses.cpu())
    return outputs.cpu()
def posesArr():
    poses = np.array([[-0.004605413803330658,1.7516345710193977]])
    return poses



if __name__ == "__main__":
    runOnCompressedData("/home/khan/modelTesting/ModelModules/preProcessing/demoData.npz")
