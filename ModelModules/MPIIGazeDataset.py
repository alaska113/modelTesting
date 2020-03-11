import torch
import torch.utils.data

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
