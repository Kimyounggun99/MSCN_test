"""
@Author: Zhi-Hao Lin
@Contact: r08942062@ntu.edu.tw
@Time: 2020/03/06
@Document: Basic operation/blocks of 3D-GCN
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 수정3 weight 1개 ##
def get_neighbor_index(vertices: "(bs, vertice_num, 3)",  neighbor_num: int): # 가장 가까운 이웃한 점 반환.
    """
    Return: (bs, vertice_num, neighbor_num)
    """
    inner = torch.bmm(vertices, vertices.transpose(1, 2)) #(bs, v, v) 행렬곱 즉, 하나의 포인트의 위치벡터와 모든 점의 위치벡터의 내적을 각각 구함
    # vertices.T(1,2): [32,3,3008]
    # inner [32, 3008, 3008]
    quadratic = torch.sum(vertices**2, dim= 2) #(bs, v) 원점으로부터 각각의 포인트가 떨어진 거리의 제곱
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
    #print(f'{distance.shape}, {vertices.shape}')
    neighbor_index = torch.topk(distance, k= neighbor_num + 1, dim= -1, largest= False)[1] # 가장 가까운 이웃 반환
    neighbor_index = neighbor_index[:, :, 1:]
    return neighbor_index

def indexing_neighbor(tensor: "(bs, vertice_num, dim)", index: "(bs, vertice_num, neighbor_num)" ):
    """
    Return: (bs, vertice_num, neighbor_num, dim) (16, 3008, neighbor_num, 3)
    """
    bs, v, n = index.size() # bs
    id_0 = torch.arange(bs).view(-1, 1, 1) #0~bs-1 (16, 1, 1)
    
    tensor_indexed = tensor[id_0, index]
    
    return tensor_indexed

def get_neighbor_direction_norm(vertices: "(bs, vertice_num, 3)", neighbor_index: "(bs, vertice_num, neighbor_num)"):
    """
    Return: (bs, vertice_num, neighobr_num, 3)
    """
    neighbors = indexing_neighbor(vertices, neighbor_index) # (bs, v, n, 3) [32,3008,3,3]
    neighbor_direction = neighbors - vertices.unsqueeze(2)
    neighbor_direction_norm = F.normalize(neighbor_direction, dim= -1)
    return neighbor_direction_norm # Direction between vertice and neighbor

# 거리값 함수
def get_neighbor_distance(vertices: "(bs, vertice_num, 3)", neighbor_index: "(bs, vertice_num, neighbor_num)"):
    """
    Return: (bs, vertice_num, neighobr_num)
    """
    neighbors = indexing_neighbor(vertices, neighbor_index) # (bs, v, n, 3)
    neighbor_direction = neighbors - vertices.unsqueeze(2)
    # Compute the Euclidean distance (magnitude of direction vector).
    neighbor_distance = torch.norm(neighbor_direction,dim=-1)
    
    return neighbor_distance


class Conv_surface(nn.Module):
    """Extract structure feafure from surface, independent from vertice coordinates"""
    def __init__(self, kernel_num, support_num):
        super().__init__()
        self.kernel_num = kernel_num
        self.support_num = support_num

        self.relu = nn.ReLU(inplace= True)
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * kernel_num)) # 커널 벡터 파라미터
        self.distance = nn.Parameter(torch.FloatTensor(1, support_num * kernel_num)) # 거리값 함수 파라미터
        self.initialize()

    def initialize(self): # 아마도 BN에 해당하는 것으로 생각.
        stdv = 1. / math.sqrt(self.support_num * self.kernel_num)
        self.directions.data.uniform_(-stdv, stdv)
        self.distance.data.uniform_(-stdv, stdv)
    
    def forward(self, 
                neighbor_index: "(bs, vertice_num, neighbor_num)", 
                vertices: "(bs, vertice_num, 3)"):
        """
        Return vertices with local feature: (bs, vertice_num, kernel_num)
        """

        ## vertices = points 임.
        bs, vertice_num, neighbor_num = neighbor_index.size()

        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index) # [32, 3008, 3, 3]
        
        distance = get_neighbor_distance(vertices, neighbor_index) # (32, 3008, 3 :이웃 포인트)
        distance = torch.max(distance, dim=2)[0]
        # distance = torch.max(distance, dim=2)

        # dis2= get_neighbor_distance_expanded(vertices, neighbor_index)
        distance = distance.contiguous().view(bs, vertice_num, 1)

        support_direction_norm = F.normalize(self.directions, dim= 0) #(3, s * k)

        

        theta = neighbor_direction_norm @ support_direction_norm # (bs, vertice_num, neighbor_num, s*k)


        distance = distance @ self.distance # (bs, vertice_num, neighbor_num, s*k)

        theta = self.relu(theta) # [32,3008,3,32]
        distance = self.relu(distance) # [32,3008,32]
        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, self.support_num, self.kernel_num) 

        distance = distance.contiguous().view(bs, vertice_num, self.support_num, self.kernel_num)

        
        theta = torch.max(theta, dim= 2)[0] # (bs, vertice_num, support_num, kernel_num) [32, 3008, 1, 32]
        
        feature= torch.sum(distance+theta, dim=2)


        return feature



class Conv_layer(nn.Module):
    def __init__(self, in_channel, out_channel, support_num):
        super().__init__()
        # arguments: 
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.support_num = support_num
        self.mlp=nn.Linear(self.out_channel*2, self.out_channel)
        # parameters:
        self.relu = nn.ReLU(inplace= True)
        self.weights = nn.Parameter(torch.FloatTensor(in_channel, (support_num + 1) * out_channel))
        self.bias = nn.Parameter(torch.FloatTensor((support_num + 1) * out_channel))
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * out_channel))
        self.distance = nn.Parameter(torch.FloatTensor(1, support_num * out_channel))   
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.out_channel * (self.support_num + 1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.directions.data.uniform_(-stdv, stdv)
        self.distance.data.uniform_(-stdv, stdv)

    def forward(self, 
                neighbor_index: "(bs, vertice_num, neighbor_index)",
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, in_channel)"):
        """
        Return: output feature map: (bs, vertice_num, out_channel)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)
        ###
        distance = get_neighbor_distance(vertices, neighbor_index) # [32, 3008, 3]
        # dis2 = torch.max(dis2, dim=2)
        distance = distance.contiguous().view(bs, vertice_num, neighbor_num, 1)
        distance = torch.max(distance, dim=2)[0]
        distance =distance.contiguous().view(bs, vertice_num, 1)


        support_direction_norm = F.normalize(self.directions, dim= 0)
        theta = neighbor_direction_norm @ support_direction_norm # (bs, vertice_num, neighbor_num, support_num * out_channel)
        
        distance = distance @ self.distance # (bs, vertice_num, neighbor_num, support_num * out_channel)
        
        theta = self.relu(theta)
        distance =self.relu(distance)

        # distance = distance.contiguous().view(bs, vertice_num, 1)

        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, -1)

       # print("theta shape:: ", theta.size())

        feature_out = feature_map @ self.weights + self.bias # (bs, vertice_num, (support_num + 1) * out_channel) 
        
        #print("feature_out shape:: ", feature_out.size())
        
        feature_center = feature_out[:, :, :self.out_channel] # (bs, vertice_num, out_channel)
        feature_support = feature_out[:, :, self.out_channel:] #(bs, vertice_num, support_num * out_channel)
        #print(":feature_center_shape::", feature_center.size() )
        #print("feature_support shape:: ", feature_support.size())
        # Fuse together - max among product
        feature_support = indexing_neighbor(feature_support, neighbor_index) # (bs, vertice_num, neighbor_num, support_num * out_channel)
        #print("feature_support shape:: ", feature_support.size())
        #### 추가 
        activation_support = theta * feature_support   # (bs, vertice_num, neighbor_num, support_num * out_channel)
        # feature_support2 = distance * feature_support2
        activation_support = activation_support.view(bs,vertice_num, neighbor_num, self.support_num, self.out_channel)
        distance = distance.view(bs,vertice_num, self.support_num, self.out_channel)
        #print("distahce!!!!!!!!!", distance.size())
        #print("activation support shape:: ", activation_support.size())
        #print("activation support shape:: ", feature_center.size())
        
        activation_support = torch.max(activation_support, dim= 2)[0] # (bs, vertice_num, support_num, out_channel)
        #print("activation_support!!!!!!!!!!!!!!: ", activation_support.size())
        activation_support = torch.sum(activation_support, dim= 2) # (bs, vertice_num, out_channel)
        #print("activation_support!!!!!!!!!!!!!!: ", activation_support.size())
        feature_fuse = feature_center + activation_support # (bs, vertice_num, out_channel)
        #print("feature_fuse shape::", feature_fuse.size())
        
        # case B
        # 여기서 concat
        #print("distance shape:: ", distance.size())
        distance= torch.sum(distance, dim=2)
        out= torch.concat([feature_fuse, distance], dim=-1)
        out= out.to(vertices.device)
        feature_fuse=self.mlp(out)
        ##print(len(feature_fuse[0,0,:])*2, self.out_channel)
        return feature_fuse



class wo_distance_Conv_surface(nn.Module):
    """Extract structure feafure from surface, independent from vertice coordinates"""
    def __init__(self, kernel_num, support_num):
        super().__init__()
        self.kernel_num = kernel_num
        self.support_num = support_num

        self.relu = nn.ReLU(inplace= True)
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * kernel_num))
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.support_num * self.kernel_num)
        self.directions.data.uniform_(-stdv, stdv)
    
    def forward(self, 
                neighbor_index: "(bs, vertice_num, neighbor_num)", 
                vertices: "(bs, vertice_num, 3)"):
        """
        Return vertices with local feature: (bs, vertice_num, kernel_num)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size() 
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)
        support_direction_norm = F.normalize(self.directions, dim= 0) #(3, s * k)
        theta = neighbor_direction_norm @ support_direction_norm # (bs, vertice_num, neighbor_num, s*k)

        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, self.support_num, self.kernel_num)
        theta = torch.max(theta, dim= 2)[0] # (bs, vertice_num, support_num, kernel_num)
        feature = torch.sum(theta, dim= 2) # (bs, vertice_num, kernel_num)
        return feature





class wo_distance_Conv_layer(nn.Module):
    def __init__(self, in_channel, out_channel, support_num):
        super().__init__()
        # arguments: 
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.support_num = support_num

        # parameters:
        self.relu = nn.ReLU(inplace= True)
        self.weights = nn.Parameter(torch.FloatTensor(in_channel, (support_num + 1) * out_channel))
        self.bias = nn.Parameter(torch.FloatTensor((support_num + 1) * out_channel))
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * out_channel))
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.out_channel * (self.support_num + 1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self, 
                neighbor_index: "(bs, vertice_num, neighbor_index)",
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, in_channel)"):
        """
        Return: output feature map: (bs, vertice_num, out_channel)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)
        support_direction_norm = F.normalize(self.directions, dim= 0)
        theta = neighbor_direction_norm @ support_direction_norm # (bs, vertice_num, neighbor_num, support_num * out_channel)
        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, -1)
        # (bs, vertice_num, neighbor_num, support_num * out_channel)

        feature_out = feature_map @ self.weights + self.bias # (bs, vertice_num, (support_num + 1) * out_channel)
        feature_center = feature_out[:, :, :self.out_channel] # (bs, vertice_num, out_channel)
        feature_support = feature_out[:, :, self.out_channel:] #(bs, vertice_num, support_num * out_channel)

        # Fuse together - max among product
        feature_support = indexing_neighbor(feature_support, neighbor_index) # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = theta * feature_support # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = activation_support.view(bs,vertice_num, neighbor_num, self.support_num, self.out_channel)
        activation_support = torch.max(activation_support, dim= 2)[0] # (bs, vertice_num, support_num, out_channel)
        activation_support = torch.sum(activation_support, dim= 2)    # (bs, vertice_num, out_channel)
        feature_fuse = feature_center + activation_support # (bs, vertice_num, out_channel)
        return feature_fuse





class Original_Conv_surface(nn.Module):
    """Extract structure feafure from surface, independent from vertice coordinates"""
    def __init__(self, kernel_num, support_num):
        super().__init__()
        self.kernel_num = kernel_num
        self.support_num = support_num

        self.relu = nn.ReLU(inplace= True)
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * kernel_num)) # 커널 벡터 파라미터
        self.distance = nn.Parameter(torch.FloatTensor(1, support_num * kernel_num)) # 거리값 함수 파라미터
        self.initialize()

    def initialize(self): # 아마도 BN에 해당하는 것으로 생각.
        stdv = 1. / math.sqrt(self.support_num * self.kernel_num)
        self.directions.data.uniform_(-stdv, stdv)
        self.distance.data.uniform_(-stdv, stdv)
    
    def forward(self, 
                neighbor_index: "(bs, vertice_num, neighbor_num)", 
                vertices: "(bs, vertice_num, 3)"):
        """
        Return vertices with local feature: (bs, vertice_num, kernel_num)
        """
        ## vertices = points 임.
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index) # [32, 3008, 3, 3]
        distance = get_neighbor_distance(vertices, neighbor_index) # (32, 3008, 3 :이웃 포인트)
        # distance = torch.max(distance, dim=2)
        max_value, max_indices = torch.max(distance, dim=2)
        distance = max_value
        distance =distance.contiguous().view(bs, vertice_num, 1)
        # dis2= get_neighbor_distance_expanded(vertices, neighbor_index)
        
        support_direction_norm = F.normalize(self.directions, dim= 0) #(3, s * k)

        

        theta = neighbor_direction_norm @ support_direction_norm # (bs, vertice_num, neighbor_num, s*k)

        distance = distance @ self.distance

        theta = self.relu(theta) # [32,3008,3,32]
        distance = self.relu(distance) # [32,3008,32]
        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, self.support_num, self.kernel_num) 

        distance = distance.contiguous().view(bs, vertice_num, self.support_num, self.kernel_num)

        theta = torch.max(theta, dim= 2)[0] # (bs, vertice_num, support_num, kernel_num) [32, 3008, 1, 32]

        feature = torch.sum(theta+distance, dim= 2) # (bs, vertice_num, kernel_num)

        return feature

class Original_Conv_layer(nn.Module):
    def __init__(self, in_channel, out_channel, support_num):
        super().__init__()
        # arguments: 
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.support_num = support_num

        # parameters:
        self.relu = nn.ReLU(inplace= True)
        self.weights = nn.Parameter(torch.FloatTensor(in_channel, (support_num + 1) * out_channel))
        self.bias = nn.Parameter(torch.FloatTensor((support_num + 1) * out_channel))
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * out_channel))
        self.distance = nn.Parameter(torch.FloatTensor(1, support_num * out_channel))
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.out_channel * (self.support_num + 1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.directions.data.uniform_(-stdv, stdv)
        self.distance.data.uniform_(-stdv, stdv)

    def forward(self, 
                neighbor_index: "(bs, vertice_num, neighbor_index)",
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, in_channel)"):
        """
        Return: output feature map: (bs, vertice_num, out_channel)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)
        ###
        distance = get_neighbor_distance(vertices, neighbor_index) # [32, 3008, 3]
        # dis2 = torch.max(dis2, dim=2)
        max_value, max_indices = torch.max(distance, dim=2)
        distance = max_value
        distance = distance.contiguous().view(bs, vertice_num, 1)

        support_direction_norm = F.normalize(self.directions, dim= 0)
        theta = neighbor_direction_norm @ support_direction_norm # (bs, vertice_num, neighbor_num, support_num * out_channel)
        
        distance = distance @ self.distance
        
        theta = self.relu(theta)
        distance =self.relu(distance)

        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, -1)

        distance = distance.contiguous().view(bs, vertice_num, -1)


        feature_out = feature_map @ self.weights + self.bias # (bs, vertice_num, (support_num + 1) * out_channel) 


        feature_center = feature_out[:, :, :self.out_channel] # (bs, vertice_num, out_channel)
        feature_support = feature_out[:, :, self.out_channel:] #(bs, vertice_num, support_num * out_channel)


        # Fuse together - max among product
        feature_support = indexing_neighbor(feature_support, neighbor_index) # (bs, vertice_num, neighbor_num, support_num * out_channel)

        #### 추가 
        activation_support = theta * feature_support   # (bs, vertice_num, neighbor_num, support_num * out_channel)
        # feature_support2 = distance * feature_support2
        activation_support = activation_support.view(bs,vertice_num, neighbor_num, self.support_num, self.out_channel)
        distance = distance.view(bs,vertice_num, self.support_num, self.out_channel)
        activation_support = torch.max(activation_support, dim= 2)[0] # (bs, vertice_num, support_num, out_channel)
        activation_support = torch.sum(activation_support+distance , dim= 2)    # (bs, vertice_num, out_channel)

        feature_fuse = feature_center + activation_support # (bs, vertice_num, out_channel)
        return feature_fuse








class Constant_Pool_layer(nn.Module):
    def __init__(self, pool_num: int= 10, neighbor_num: int=  4):
        super().__init__()
        self.pool_num = pool_num
        self.neighbor_num = neighbor_num

    def forward(self, 
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, channel_num)"):
        """
        Return:
            vertices_pool: (bs, pool_vertice_num, 3),
            feature_map_pool: (bs, pool_vertice_num, channel_num)
        """
        bs, vertice_num, _ = vertices.size()
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        neighbor_feature = indexing_neighbor(feature_map, neighbor_index) #(bs, vertice_num, neighbor_num, channel_num)
        pooled_feature = torch.max(neighbor_feature, dim= 2)[0] #(bs, vertice_num, channel_num)

        sample_idx = torch.randperm(vertice_num)[:self.pool_num]
        vertices_pool = vertices[:, sample_idx, :] # (bs, pool_num, 3)
        feature_map_pool = pooled_feature[:, sample_idx, :] #(bs, pool_num, channel_num)
        return vertices_pool, feature_map_pool

class Pool_layer(nn.Module):
    def __init__(self, pooling_rate: int= 4, neighbor_num: int= 4):
        super().__init__()
        self.pooling_rate = pooling_rate
        self.neighbor_num= neighbor_num
    def forward(self, 
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, channel_num)"):
        """
        Return:
            vertices_pool: (bs, pool_vertice_num, 3),
            feature_map_pool: (bs, pool_vertice_num, channel_num)
        """
        bs, vertice_num, _ = vertices.size()
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        neighbor_feature = indexing_neighbor(feature_map, neighbor_index) #(bs, vertice_num, neighbor_num, channel_num)
        pooled_feature = torch.max(neighbor_feature, dim= 2)[0] #(bs, vertice_num, channel_num)

        pool_num = int(vertice_num / self.pooling_rate)
        sample_idx = torch.randperm(vertice_num)[:pool_num]
        vertices_pool = vertices[:, sample_idx, :] # (bs, pool_num, 3)
        feature_map_pool = pooled_feature[:, sample_idx, :] #(bs, pool_num, channel_num)
        return vertices_pool, feature_map_pool



def test():
    import time
    bs = 16
    v = 3008
    dim = 3
    n = 3
    vertices = torch.randn(bs, v, dim)
    neighbor_index = get_neighbor_index(vertices, n)

    s = 1
    conv_1 = Conv_surface(kernel_num= 32, support_num= s)
    conv_2 = Conv_layer(in_channel= 32, out_channel= 64, support_num= s)
    
    pool = Pool_layer(pooling_rate= 4, neighbor_num= 4)
    
    print("Input size: {}".format(vertices.size()))
    start = time.time()
    f1 = conv_1(neighbor_index, vertices)
    print("\n[1] Time: {}".format(time.time() - start))
    print("[1] Out shape: {}".format(f1.size()))
    start = time.time()
    f2 = conv_2(neighbor_index, vertices, f1)
    print("\n[2] Time: {}".format(time.time() - start))
    print("[2] Out shape: {}".format(f2.size()))
    start = time.time()
    v_pool, f_pool = pool(vertices, f2)
    print("\n[3] Time: {}".format(time.time() - start))
    print("[3] v shape: {}, f shape: {}".format(v_pool.size(), f_pool.size()))


if __name__ == "__main__":
    test()

