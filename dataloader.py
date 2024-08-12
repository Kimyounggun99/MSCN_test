import os
import sys
import glob
import h5py
import numpy as np

import pypotree 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import sklearn.metrics as metrics

def load_data(partition):
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join('modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            
            np.random.shuffle(pointcloud)
            
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


def main():
    import argparse
    import open3d as o3d

    parser= argparse.ArgumentParser()

    parser.add_argument('-mode', default= 'get_pcd')
    parser.add_argument('-save_transfered_img', default= None, help= 'Path and model name to save image')
    args= parser.parse_args()

    if args.save_transfered_img is not None:

        transfered_img_path= args.save_transfered_img
    else:
        transfered_img_path= '/home/mlmlab13/Domain_generalization/yg/ModelNet_test/saved_pcd_img'
    if args.mode== 'get_pcd':
        
        modelnet40 = ModelNet40(2048)
        idx = 60
        points, label = modelnet40.__getitem__(idx)
        #breakpoint()
        label_to_name = np.loadtxt('modelnet40_ply_hdf5_2048/shape_names.txt', dtype=str)
        label_to_name[label]
        breakpoint()
        #pc_path = pypotree.generate_cloud_for_display(points)
        #pypotree.display_cloud_colab(pc_path)
        #pc_path.add_points(points)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(os.path.join(transfered_img_path, 'modelnet_transfered_image.ply'), point_cloud)


        print("end")
if __name__ == "__main__":
    main()