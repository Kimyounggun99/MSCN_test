
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gcn3d import Conv_surface, Conv_layer, Pool_layer, wo_distance_Conv_layer, get_neighbor_index
from model_gcn import MSCN
#from model_dgcnn import PointNetfeat, PointNetCls
import math
import copy

class Rand_ADAIN3D(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        
        self.fc = nn.Linear(style_dim, num_features*2)
        self.num_features= num_features

    def forward(self, x, s): 
        h = self.fc(s) # (bs, 128)
        mean= torch.mean(x, dim=1, keepdim=True)
        std= torch.std(x, dim=1, unbiased=True, keepdim=True) + 1e-8
        #breakpoint()
        x= (x-mean)/std #(bs, pts, dim)
        gamma, beta = torch.chunk(h, chunks=2, dim=-1) #(bs, 64) (bs, 64)

        return (1 + gamma) * x + beta
    

class Autoencoder(nn.Module):
    def __init__(self, model, support_num:int, neighbor_num: int, args, use_target_domain=False):
        super().__init__()

        """
        Input shape: (bs, num points input_ channel)

        output shape: (bs, num points, output channel)
        """
        self.device= torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
        self.support_num= support_num
        self.neighbor_num= neighbor_num
        self.use_target_domain= use_target_domain
        self.args= args
        self.style_dim= 10

        self.copy_model= copy.deepcopy(model)
        self.mse_loss= nn.MSELoss()


        self.deconv1= Conv_layer(32, 16, support_num)
        self.deconv2= Conv_layer(16, 3, support_num)
        self.adain= Rand_ADAIN3D(self.style_dim, 32).to(self.device)


        

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean= torch.mean(input, dim=1, keepdim=True)
        input_std= torch.std(input, dim=1, unbiased=True, keepdim=True) + 1e-8
        target_mean= torch.mean(target, dim=1, keepdim=True)
        target_std= torch.std(target, dim=1, unbiased=True, keepdim=True) + 1e-8

        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, source, alpha=1.0):


       
        with torch.no_grad(): 
            neighbor_index = get_neighbor_index(source, self.neighbor_num)
            source_feature1= self.copy_model.conv_0(neighbor_index, source)
            source_feature1= F.relu(source_feature1, inplace=True)
            source_feature2= self.copy_model.conv_1(neighbor_index, source, source_feature1)
            source_feature2= F.relu(source_feature2, inplace=True)
            
            target_feature= torch.randn(len(source_feature2), len(source_feature2[1]), self.style_dim)
            target_feature= target_feature.to(self.device)
            
            t= self.adain(source_feature2, target_feature)
        construct_feature1= self.deconv1(neighbor_index, source, t)
        construct_feature1= F.relu(construct_feature1, inplace=True)
        construct_feature2= self.deconv2(neighbor_index, source, construct_feature1)
        construct_feature2= torch.sigmoid(construct_feature2)
        gen_points= construct_feature2
            

        return gen_points
        
        
            



def test():
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default= 'train', help= '[train/test]')
    parser.add_argument('-model', default= 'pointnet', help= '[pointnet/dgcnn/gcn3d]')
    parser.add_argument('-cuda', default= '4', help= 'Cuda index')
    parser.add_argument('-lr', type= float, default= 1e-4, help= 'Learning rate')
    parser.add_argument('-bs', type= int, default= 16, help= 'Batch size')
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
    parser.add_argument('-gen_name', type=str, default='adain', help='domain generator')
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

    parser.add_argument('-w_style', type=float, default=10.0)
    parser.add_argument('-w_content', type=float, default=1.0)
    
    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

    bs = 16
    v = 3008
    dim = 3
    n = 3
    s = 1
    source = torch.randn(bs, v, dim)
    target = torch.randn(bs, v ,dim)
    
    import numpy as np

    #a= np.array([[1,2,3],[4,5,6],[7,8,9]])
    #b= np.array([[1,2,3]])
#
    #breakpoint()
    
   

    if args.model== 'scn':
        model= ASCN(support_num= args.support, neighbor_num= args.neighbor)
        path= '/home/mlmlab13/Domain_generalization/yg/saved_model/pre_trained_model/scn_kitti_model_ep20.pkl'
    elif args.model== 'dgcnn':
        model= DGCNN()
        path= '/home/mlmlab13/Domain_generalization/yg/saved_model/pre_trained_model/dgcnn_kitti_model_ep20.pkl'
    elif args.model== 'pointnet':
        model= PointNetCls(3)
        path= '/home/mlmlab13/Domain_generalization/yg/saved_model/pre_trained_model/pointnet_kitti_model_ep20.pkl'

    model.load_state_dict(torch.load(path)['cls_net'])

    source= source.to(device)
    target= target.to(device)

    filter= Autoencoder(model, s, n, args, use_target_domain=True)


    generator= Autoencoder(model, s, n, args, use_target_domain=False)
    

    filter= filter.to(device)
    generator= generator.to(device)
    generator(source)
    filter(source, target, validation_mode=True)
    filter(source, validation_mode=True)

if __name__=="__main__":
    test()