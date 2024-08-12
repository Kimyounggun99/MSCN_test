import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import sys
import gcn3d



class GCN(nn.Module):
    def __init__(self, support_num: int, neighbor_num: int, class_num= 3):
        super().__init__()
        self.neighbor_num = neighbor_num

        self.conv_0 = gcn3d.wo_distance_Conv_surface(kernel_num= 32, support_num= support_num)
        self.conv_1 = gcn3d.wo_distance_Conv_layer(32, 64, support_num= support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = gcn3d.wo_distance_Conv_layer(64, 128, support_num= support_num)
        self.conv_3 = gcn3d.wo_distance_Conv_layer(128, 256, support_num= support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = gcn3d.wo_distance_Conv_layer(256, 1024, support_num= support_num)



        self.classifier = nn.Sequential(
            nn.Linear(1024, 256), 
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace= False),
            nn.Linear(256, class_num)
        )
        self.projection = nn.Sequential(
            nn.Linear(1024, 256), 
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace= False),
            nn.Linear(256, 128)
        )

    def forward(self,  vertices: "(bs, vertice_num, 3)"):
        bs, vertice_num, _ = vertices.size()
        
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num) # Get neighbor index per each point(bs, vertice_num, neighbor_num)
        fm_0 = self.conv_0(neighbor_index, vertices) # Extract surface(local) feature, vertices with local feature: (bs, vertice_num, kernel_num)
        fm_0 = F.relu(fm_0, inplace= True)
        fm_1 = self.conv_1(neighbor_index, vertices, fm_0) # feature map: (bs, vertice_num, out_channel)

        #print("fm_1 shape:::: ", fm_1.size())

        fm_1 = F.relu(fm_1, inplace= True)
        vertices, fm_1 = self.pool_1(vertices, fm_1) #vertices_pool: (bs, pool_vertice_num, 3),feature_map_pool: (bs, pool_vertice_num, channel_num)

        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        fm_2 = self.conv_2(neighbor_index, vertices, fm_1)
        fm_2 = F.relu(fm_2, inplace= True)

        fm_3 = self.conv_3(neighbor_index, vertices, fm_2)
        fm_3 = F.relu(fm_3, inplace= True) 
        vertices, fm_3 = self.pool_2(vertices, fm_3)

        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        fm_4 = self.conv_4(neighbor_index, vertices, fm_3) # (bs, vertices_num, 1024)
        feature_global = fm_4.max(1)[0] 
        
        
        class_output = self.classifier(feature_global)
        hidden_output= self.projection(feature_global)
        hidden_output= F.normalize(hidden_output)
        return feature_global, class_output, hidden_output





class MSCN_with_only_SCL(nn.Module):
    def __init__(self, support_num: int, neighbor_num: int, class_num=3):
        super().__init__()
        self.neighbor_num = neighbor_num

        self.conv_0 = gcn3d.Conv_surface(kernel_num= 32, support_num= support_num)
        self.conv_1 = gcn3d.Conv_layer(32, 64, support_num= support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = gcn3d.Conv_layer(64, 128, support_num= support_num)
        self.conv_3 = gcn3d.Conv_layer(128, 256, support_num= support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = gcn3d.Conv_layer(256, 1024, support_num= support_num)



        self.classifier = nn.Sequential(
            nn.Linear(1024, 256), 
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace= False),
            nn.Linear(256, class_num)
        )
        self.projection = nn.Sequential(
            nn.Linear(1024, 256), 
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace= False),
            nn.Linear(256, 128)
        )

    def forward(self,  vertices: "(bs, vertice_num, 3)"):
        bs, vertice_num, _ = vertices.size()
        
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num) # Get neighbor index per each point(bs, vertice_num, neighbor_num)
        fm_0 = self.conv_0(neighbor_index, vertices) # Extract surface(local) feature, vertices with local feature: (bs, vertice_num, kernel_num)
        fm_0 = F.relu(fm_0, inplace= True)
        fm_1 = self.conv_1(neighbor_index, vertices, fm_0) # feature map: (bs, vertice_num, out_channel)

        #print("fm_1 shape:::: ", fm_1.size())

        fm_1 = F.relu(fm_1, inplace= True)
        vertices, fm_1 = self.pool_1(vertices, fm_1) #vertices_pool: (bs, pool_vertice_num, 3),feature_map_pool: (bs, pool_vertice_num, channel_num)

        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        fm_2 = self.conv_2(neighbor_index, vertices, fm_1)
        fm_2 = F.relu(fm_2, inplace= True)

        fm_3 = self.conv_3(neighbor_index, vertices, fm_2)
        fm_3 = F.relu(fm_3, inplace= True) 
        vertices, fm_3 = self.pool_2(vertices, fm_3)

        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        fm_4 = self.conv_4(neighbor_index, vertices, fm_3) # (bs, vertices_num, 1024)
        feature_global = fm_4.max(1)[0] 
        
        
        class_output = self.classifier(feature_global)
        hidden_output= self.projection(feature_global)
        hidden_output= F.normalize(hidden_output)
        return feature_global, class_output, hidden_output



class SCN(nn.Module):
    def __init__(self, support_num: int, neighbor_num: int, class_num=3):
        super().__init__()
        self.neighbor_num = neighbor_num

        self.conv_0 = gcn3d.Original_Conv_surface(kernel_num= 32, support_num= support_num)
        self.conv_1 = gcn3d.Original_Conv_layer(32, 64, support_num= support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = gcn3d.Original_Conv_layer(64, 128, support_num= support_num)
        self.conv_3 = gcn3d.Original_Conv_layer(128, 256, support_num= support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = gcn3d.Original_Conv_layer(256, 1024, support_num= support_num)



        self.classifier = nn.Sequential(
            nn.Linear(1024, 256), 
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace= False),
            nn.Linear(256, class_num)
        )
        self.projection = nn.Sequential(
            nn.Linear(1024, 256), 
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace= False),
            nn.Linear(256, 128)
        )

    def forward(self,  vertices: "(bs, vertice_num, 3)"):
        bs, vertice_num, _ = vertices.size()
        
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num) # Get neighbor index per each point(bs, vertice_num, neighbor_num)
        fm_0 = self.conv_0(neighbor_index, vertices) # Extract surface(local) feature, vertices with local feature: (bs, vertice_num, kernel_num)
        fm_0 = F.relu(fm_0, inplace= True)
        fm_1 = self.conv_1(neighbor_index, vertices, fm_0) # feature map: (bs, vertice_num, out_channel)

        #print("fm_1 shape:::: ", fm_1.size())

        fm_1 = F.relu(fm_1, inplace= True)
        vertices, fm_1 = self.pool_1(vertices, fm_1) #vertices_pool: (bs, pool_vertice_num, 3),feature_map_pool: (bs, pool_vertice_num, channel_num)

        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        fm_2 = self.conv_2(neighbor_index, vertices, fm_1)
        fm_2 = F.relu(fm_2, inplace= True)

        fm_3 = self.conv_3(neighbor_index, vertices, fm_2)
        fm_3 = F.relu(fm_3, inplace= True) 
        vertices, fm_3 = self.pool_2(vertices, fm_3)

        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        fm_4 = self.conv_4(neighbor_index, vertices, fm_3) # (bs, vertices_num, 1024)
        feature_global = fm_4.max(1)[0] 
        
        
        class_output = self.classifier(feature_global)
        hidden_output= self.projection(feature_global)
        hidden_output= F.normalize(hidden_output)
        return feature_global, class_output, hidden_output



class MSCN(nn.Module):
    def __init__(self, support_num: int, neighbor_num: int, class_num=3):
        super().__init__()
        self.neighbor_num = neighbor_num

        self.conv_0 = gcn3d.Conv_surface(kernel_num= 16, support_num= support_num)
        self.conv_1 = gcn3d.Conv_layer(16, 32, support_num= support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)

        self.conv_2 = gcn3d.Conv_layer(32, 64, support_num= support_num)
        self.conv_3= gcn3d.Conv_layer(128, 256, support_num= support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_4= gcn3d.Conv_layer(256, 256, support_num= support_num)
        self.conv_5= gcn3d.Conv_layer(512, 1024, support_num= support_num)



        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),  # Layer추가 실험하기
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace= False),
            nn.Linear(256, class_num)
        )
        self.projection = nn.Sequential(
            nn.Linear(1024, 256), 
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace= False),
            nn.Linear(256, 128)
        )

    def forward(self,  vertices: "(bs, vertice_num, 3)"):
        bs, vertice_num, _ = vertices.size()
                
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num) # Get neighbor index per each point(bs, vertice_num, neighbor_num)
        fm_0 = self.conv_0(neighbor_index, vertices) # Extract surface(local) feature, vertices with local feature: (bs, vertice_num, kernel_num)
        fm_0 = F.relu(fm_0, inplace= True)
        #max_pool_fm0= torch.max(fm_0, dim=1)[0].view(bs,1,len(fm_0[0,0])).expand(bs,vertice_num,len(fm_0[0,0]))
        #fm_0= torch.concat((fm_0, max_pool_fm0), dim=-1)

        fm_1 = self.conv_1(neighbor_index, vertices, fm_0) # feature map: (bs, vertice_num, out_channel)
        fm_1 = F.relu(fm_1, inplace= True)
        vertices, fm_1= self.pool_1(vertices, fm_1)
        bs, vertice_num, _ = vertices.size()
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        
        fm_2 = self.conv_2(neighbor_index, vertices, fm_1)
        fm_2 = F.relu(fm_2, inplace= True)

        max_pool_fm_2= torch.max(fm_2, dim=1)[0].view(bs,1,len(fm_2[0,0])).expand(bs,vertice_num,len(fm_2[0,0]))
        fm_2= torch.concat((fm_2, max_pool_fm_2), dim=-1)

        #max_pool_fm_2= torch.max(fm_2, dim=1)[0].view(bs,1,len(fm_2[0,0])).expand(bs,vertice_num,len(fm_2[0,0]))
        #fm_2= torch.concat((fm_2, max_pool_fm_2), dim=-1)
        fm_3 = self.conv_3(neighbor_index, vertices, fm_2)
        fm_3 = F.relu(fm_3, inplace= True) 
        
        vertices, fm_3= self.pool_2(vertices, fm_3)
        bs, vertice_num, _ = vertices.size()
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        
        fm_4 = self.conv_4(neighbor_index, vertices, fm_3)
        fm_4 = F.relu(fm_4, inplace= True)
        
        max_pool_fm_4= torch.max(fm_4, dim=1)[0].view(bs,1,len(fm_4[0,0])).expand(bs,vertice_num,len(fm_4[0,0]))
        fm_4= torch.concat((fm_4, max_pool_fm_4), dim=-1)

        fm_5 = self.conv_5(neighbor_index, vertices, fm_4)
        fm_5 = F.relu(fm_5, inplace= True)
        #breakpoint()
        feature_global = fm_5.max(1)[0] 
        #feature_global= fm_3.mean(1)
        #breakpoint()
        class_output = self.classifier(feature_global)
        hidden_output= self.projection(feature_global)
        hidden_output= F.normalize(hidden_output)
        return feature_global, class_output, hidden_output







class SCN_without_pool(nn.Module):
    def __init__(self, support_num: int, neighbor_num: int):
        super().__init__()
        self.neighbor_num = neighbor_num

        self.conv_0 = gcn3d.Conv_surface(kernel_num= 32, support_num= support_num)
        self.conv_1 = gcn3d.Conv_layer(32, 64, support_num= support_num)
        self.conv_2 = gcn3d.Conv_layer(64, 128, support_num= support_num)
        self.conv_3 = gcn3d.Conv_layer(128, 256, support_num= support_num)
        self.conv_4 = gcn3d.Conv_layer(256, 1024, support_num= support_num)
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256), 
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace= False),
            nn.Linear(256, 3)
        )
        self.projection = nn.Sequential(
            nn.Linear(1024, 256), 
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace= False),
            nn.Linear(256, 128)
        )

    def forward(self,  vertices: "(bs, vertice_num, 3)"):
        bs, vertice_num, _ = vertices.size()
        
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num) # Get neighbor index per each point(bs, vertice_num, neighbor_num)
        fm_0 = self.conv_0(neighbor_index, vertices) # Extract surface(local) feature, vertices with local feature: (bs, vertice_num, kernel_num)
        fm_0 = F.relu(fm_0, inplace= True)
        fm_1 = self.conv_1(neighbor_index, vertices, fm_0) # feature map: (bs, vertice_num, out_channel)

        #print("fm_1 shape:::: ", fm_1.size())

        fm_1 = F.relu(fm_1, inplace= True)

        fm_2 = self.conv_2(neighbor_index, vertices, fm_1)
        fm_2 = F.relu(fm_2, inplace= True)

        fm_3 = self.conv_3(neighbor_index, vertices, fm_2)
        fm_3 = F.relu(fm_3, inplace= True) 

        fm_4 = self.conv_4(neighbor_index, vertices, fm_3) # (bs, vertices_num, 1024)
        feature_global = fm_4.max(1)[0] 
        
        class_output = self.classifier(feature_global)
        hidden_output= self.projection(feature_global)
        hidden_output= F.normalize(hidden_output)
        return feature_global, class_output, hidden_output
        








class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class MSCNDANN(nn.Module):
    def __init__(self, support_num: int, neighbor_num: int, class_num=3):
        super().__init__()
        self.neighbor_num = neighbor_num

        self.conv_0 = gcn3d.Conv_surface(kernel_num= 16, support_num= support_num)
        self.conv_1 = gcn3d.Conv_layer(16, 32, support_num= support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)

        self.conv_2 = gcn3d.Conv_layer(32, 64, support_num= support_num)
        self.conv_3= gcn3d.Conv_layer(128, 256, support_num= support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_4= gcn3d.Conv_layer(256, 256, support_num= support_num)
        self.conv_5= gcn3d.Conv_layer(512, 1024, support_num= support_num)



        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),  # Layer추가 실험하기
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace= False),
            nn.Linear(256, class_num),
            nn.LogSoftmax(dim=1)
        )
        
        self.domain_classifier= nn.Sequential(
            nn.Linear(1024, 256),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace= True),
            nn.Linear(256, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self,  vertices: "(bs, vertice_num, 3)", alpha):
        bs, vertice_num, _ = vertices.size()
                
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num) # Get neighbor index per each point(bs, vertice_num, neighbor_num)
        fm_0 = self.conv_0(neighbor_index, vertices) # Extract surface(local) feature, vertices with local feature: (bs, vertice_num, kernel_num)
        fm_0 = F.relu(fm_0, inplace= True)
        #max_pool_fm0= torch.max(fm_0, dim=1)[0].view(bs,1,len(fm_0[0,0])).expand(bs,vertice_num,len(fm_0[0,0]))
        #fm_0= torch.concat((fm_0, max_pool_fm0), dim=-1)

        fm_1 = self.conv_1(neighbor_index, vertices, fm_0) # feature map: (bs, vertice_num, out_channel)
        fm_1 = F.relu(fm_1, inplace= True)
        vertices, fm_1= self.pool_1(vertices, fm_1)
        bs, vertice_num, _ = vertices.size()
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        
        fm_2 = self.conv_2(neighbor_index, vertices, fm_1)
        fm_2 = F.relu(fm_2, inplace= True)

        max_pool_fm_2= torch.max(fm_2, dim=1)[0].view(bs,1,len(fm_2[0,0])).expand(bs,vertice_num,len(fm_2[0,0]))
        fm_2= torch.concat((fm_2, max_pool_fm_2), dim=-1)

        #max_pool_fm_2= torch.max(fm_2, dim=1)[0].view(bs,1,len(fm_2[0,0])).expand(bs,vertice_num,len(fm_2[0,0]))
        #fm_2= torch.concat((fm_2, max_pool_fm_2), dim=-1)
        fm_3 = self.conv_3(neighbor_index, vertices, fm_2)
        fm_3 = F.relu(fm_3, inplace= True) 
        
        vertices, fm_3= self.pool_2(vertices, fm_3)
        bs, vertice_num, _ = vertices.size()
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        
        fm_4 = self.conv_4(neighbor_index, vertices, fm_3)
        fm_4 = F.relu(fm_4, inplace= True)
        
        max_pool_fm_4= torch.max(fm_4, dim=1)[0].view(bs,1,len(fm_4[0,0])).expand(bs,vertice_num,len(fm_4[0,0]))
        fm_4= torch.concat((fm_4, max_pool_fm_4), dim=-1)

        fm_5 = self.conv_5(neighbor_index, vertices, fm_4)
        fm_5 = F.relu(fm_5, inplace= True)
        #breakpoint()
        feature_global = fm_5.max(1)[0] 
        
        #print("Featrues were extracted successfully!!!!")
        #print("Features's shape ::: ", feature_global.size())
        reverse_feature = ReverseLayerF.apply(feature_global, alpha)
        #print("Reverse featrues were extracted successfully!!!!")
        #print("Reverse feature's shape::: ", reverse_feature.size())


        class_output = self.classifier(feature_global)
        domain_output= self.domain_classifier(reverse_feature)
        
        return class_output, domain_output








def test():
    import time
    sys.path.append("../")
    from util import parameter_number
    
    device = torch.device('cuda:4')
    points = torch.zeros(32, 3008, 3).to(device)
    model = SCN_with_interval_maxpooling(support_num= 1, neighbor_num= 1).to(device)
    start = time.time()
    _, output, _ = model(points)
    breakpoint()
    print("Inference time: {}".format(time.time() - start))
    print("Parameter #: {}".format(parameter_number(model)))
    print("Inputs size: {}".format(points.size()))
    print("Output size: {}".format(output.size()))

if __name__ == '__main__':
    test()

