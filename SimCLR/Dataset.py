
import cv2
from torch.utils.data import Dataset
from pathlib import Path

class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]
    
class SemanticKITTI(Dataset):
    def __init__(self, split="train", transforms=None):
        dataset_path = "/home/user/SemanticKitti/dataset/sequences/"
        seqs = {
        "train" : [
        '00',
        '02',
        '03',
        '04',
        '05',
        '06',
        '09',
        '10'],
        "val" : [ '07', '01'],
        "test" : [ '08'],
        }
        self.split = split
        self.data_paths = []
        for seq in seqs[split]:
            self.data_paths.extend(list((Path(dataset_path) / seq).rglob("**/image_2/*.png")))
        self.data_paths = sorted(self.data_paths)
        self.transforms = transforms
        self.i = 0
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index):
        img_path = self.data_paths[index]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return self.transforms(img)
        