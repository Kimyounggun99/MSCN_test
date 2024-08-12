import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from torchvision import models

import argparse
from torch.utils.data import DataLoader
import sys
import pandas as pd
import time
import datetime

from dataset_modelnet import ModelNet_pointcloud
from domain_gen_manager import Manager
from util import Transform
from model_gcn import MSCN


from con_losses import SupConLoss


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default= 'train', help= '[train/test]')
    parser.add_argument('-model', default= 'mscn', help= '[pointnet/dgcnn/gcn3d]')
    parser.add_argument('-classnum', type=int, default= 3, help= '[pointnet/dgcnn/gcn3d]')
    parser.add_argument('-cuda', default= '0', help= 'Cuda index')
    parser.add_argument('-lr', type= float, default= 1e-4, help= 'Learning rate')
    parser.add_argument('-bs', type= int, default= 4, help= 'Batch size')
    parser.add_argument('-dataset', help= "Path to modelnet point cloud data")
    parser.add_argument('-load', help= 'Path to load model')
    parser.add_argument('-save', default= None, help= 'Path and model name to save model')
    parser.add_argument('-save_img', default= None, help= '0/1')
    parser.add_argument('-record', help= 'Record file name (e.g. record.log)')
    parser.add_argument('-interval', type= int, default= 120, help= 'Record interval within an epoch')
    parser.add_argument('-support', type= int, default= 1, help= 'Support number')
    parser.add_argument('-neighbor', type= int, default= 3, help= 'Neighbor number')
    parser.add_argument('-normal', dest= 'normal', action= 'store_true', help= 'Normalize objects (zero-mean, unit size)')
    parser.set_defaults(normal= False)
    parser.add_argument('-shift', type= float, help= 'Shift objects (original: 0.0)')
    parser.add_argument('-scale', type= float, help= 'Enlarge/shrink objects (original: 1.0)')
    parser.add_argument('-rotate', type= float, help= 'Rotate objects in degree (original: 0.0)')
    parser.add_argument('-axis', type= int, default= 2, help= 'Rotation axis [0, 1, 2] (upward = 2)')
    parser.add_argument('-random', dest= 'random', action= 'store_true', help= 'Randomly transform in a given range')
    parser.set_defaults(random= False)
    parser.add_argument('-gen_name', type=str, default='auto', help='cnn/hr')
    parser.add_argument('-n_tgt', type=int, default=10, help='the number of generated domain')
    parser.add_argument('-tgt_epochs', type= int, default= 80, help= 'Epoch number')
    parser.add_argument('-tgt_epochs_fixg', type=int, default=None, help='epoch threshold')
    parser.add_argument('-w_cls', type=float, default=1.0, help='cls weight')
    parser.add_argument('-w_info', type=float, default=1.0, help='adv weight')
    parser.add_argument('-w_cyc', type=float, default=10.0, help='cycle weight')
    parser.add_argument('-w_div', type=float, default=1.0, help='divergence weight')
    parser.add_argument('-div_thresh', type=float, default=0.1, help='div_loss threshold')
    parser.add_argument('-w_tgt', type=float, default=1.0, help='generated domain weight')
    parser.add_argument('-max_tgt', type=float, default=1.0, help='generated domain weight')
    args = parser.parse_args()

    
    model= MSCN(support_num= args.support, neighbor_num= args.neighbor, class_num= args.classnum)

    manager= Manager(model, args)
    

    transform = Transform(
        normal= args.normal,
        shift= args.shift,
        scale= args.scale,
        rotate= args.rotate,
        axis= args.axis,
        random= args.random
    )



    if args.mode == "train":
        print('Trianing ...')
        print("dataset path:: ", args.dataset)
        train_data = ModelNet_pointcloud(args.dataset, 'train', transform= transform)
        train_loader = DataLoader(train_data, shuffle= True, batch_size= args.bs) # 수정 drop_last= True 추가

        test_data = ModelNet_pointcloud(args.dataset, 'test', transform= transform)
        
        test_loader = DataLoader(test_data, shuffle= False, batch_size= args.bs)
        
        manager.train(train_loader, test_loader)
       

    else:
        start =time.time()
        print(start)
        print('Testing ...')
        print("test dataset:: ", args.dataset)
        print("test_model:: ", args.model)
 
        test_data = ModelNet_pointcloud(args.dataset, 'test', transform= transform)
        test_loader = DataLoader(test_data, shuffle= False, batch_size= args.bs)
        
        test_loss, test_acc = manager.test(test_loader) 
        
        print('Test Acc:  {:.5f}'.format(test_acc))

        sec = time.time()-start

        print(sec)
        
if __name__ == '__main__':
    main()