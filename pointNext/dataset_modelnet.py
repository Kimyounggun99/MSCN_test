import os
import torch
from torch.utils.data import Dataset, DataLoader
from pyntcloud import PyntCloud
import torch.nn.functional as F
import numpy as np

class ModelNet_pointcloud(Dataset):
    # IROS에서 조금씩 변화시키면서 얻은 교훈
    # 1. dataset을 모두 positive로 이동시키면 안됨.
    
    def __init__(self, dataset_path, mode= 'train', category= None, transform= None):
        super().__init__()
        self.transform = transform

        categories_all = [name for name in os.listdir(dataset_path) if name not in ['.DS_Store', 'README.txt']]
        #print(categories_all)
        if category:
            if category not in categories_all:
                raise Exception('Categoty not found !')
            self.categories = [category]
        else:
            self.categories = categories_all
        
        self.categories.sort()
        self.path_label_pairs = []
        
        data_size_criterion=1000000


        for category in self.categories:
            label = categories_all.index(category)
            folder_path = os.path.join(dataset_path, category, mode)
            pairs = [(os.path.join(folder_path, name), label) for name in os.listdir(folder_path) if name != '.DS_Store']
            data_size_criterion= min(data_size_criterion, len(pairs))

        data_size_criterion*=12 # maximum distribution ratio 12:1 
        for category in self.categories:
            label = categories_all.index(category)
            folder_path = os.path.join(dataset_path, category, mode)
            pairs = [(os.path.join(folder_path, name), label) for name in os.listdir(folder_path) if name != '.DS_Store']
            #print("before pair",len(pairs))
            if len(pairs)>data_size_criterion:
                pairs= pairs[:data_size_criterion]
            #print("After pairs",len(pairs))
            self.path_label_pairs += pairs



    def __len__(self):
        return len(self.path_label_pairs)

    # def __getitem__(self, index):
    #     path, label = self.path_label_pairs[index]
    #     label = torch.LongTensor([label])
    #     obj = PyntCloud.from_file(path)
    #     points = torch.FloatTensor(obj.xyz)
    #     if self.transform:
    #         points = self.transform(points)
    #     return points, label

    def __getitem__(self, index):
        path, label = self.path_label_pairs[index]
        label = torch.LongTensor([label])
        obj = PyntCloud.from_file(path)
        origin_points = torch.FloatTensor(obj.xyz)
       
        # with open("paths.txt", "a") as file:
        #     file.write(path + "\n")
     
        # 코드 수정 시작.
        # Pad or truncate the points tensors to a fixed size
        desired_size = (3008, 3)  # Set your desired size here
        #breakpoint()
        pcd_len= len(origin_points)
        if len(origin_points)< desired_size[0]:
            pcd_len= len(origin_points)
        else:
            pcd_len= desired_size[0]

        current_size = origin_points.size()
        min_val_point= torch.min(origin_points, dim=0)[0]
        #breakpoint()
        #origin_points= origin_points+abs(min_val_point)
        
        if current_size[0] < desired_size[0]:
            padding = torch.zeros(desired_size[0] - current_size[0], current_size[1])
            origin_points = torch.cat([origin_points, padding])
        elif current_size[0] >= desired_size[0]:
            origin_points = origin_points[:desired_size[0], :]
        # 코드 수정 마지막 (원래는 이 부분이 아예 없었음.)

        if self.transform:
            origin_points = self.transform(origin_points)
        # print(points)
        #breakpoint()
        return origin_points, label



def test():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-source')
    parser.add_argument('-bs', type= int, default= 8)
    args = parser.parse_args()
    
    path= '/home/mlmlab13/3dgcn/classification/dataset/kitti_3'

    dataset = ModelNet_pointcloud(path, mode= "train")
    print(len(dataset))
    #print("# of Data:", len(dataset))
    dataloader = DataLoader(dataset, batch_size= args.bs) #drop_last= True, collate_fn = my_collate_fn
    for i, (points, labels) in enumerate(dataloader):    
        print(points.size())
        print(labels.size())
        break

if __name__ == '__main__':
    test()