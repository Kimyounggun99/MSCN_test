import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x= x.to('cpu')
        iden= iden.to('cpu')
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False, args=None):
        super(PointNetfeat, self).__init__()

        self.device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x, with_intermediate= False):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x= x.to(self.device)
        trans= trans.to(self.device)
        x = torch.bmm(x, trans)
        
        x = x.transpose(2, 1)
        x= x.to(self.device)
        feat1 = F.relu(self.bn1(self.conv1(x))) #(bs, 64, 3008)


        if self.feature_transform:
            trans_feat = self.fstn(feat1)
            feat1 = feat1.transpose(2,1)
            feat1 = torch.bmm(feat1, trans_feat)
            feat1 = feat1.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = feat1
        feat2 = F.relu(self.bn2(self.conv2(feat1))) # (bs, pts, 128)
        feat3 = self.bn3(self.conv3(feat2)) #(bs, pts, 1024)
        global_feat = torch.max(feat3, 2, keepdim=True)[0]
        global_feat = global_feat.view(-1, 1024)
        

        if with_intermediate:
            return feat1, feat2, feat3, global_feat

        if self.global_feat:
            return global_feat, trans, trans_feat
        else:
            global_feat = global_feat.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([global_feat, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=3, feature_transform=False, args=None):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform

        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform, args=args)
        
        self.classifier= nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k)
        )

        self.projection= nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)



        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x, with_intermediate=False):
        x = x.transpose(1, 2)
        if with_intermediate:
            feat1, feat2, feat3, gl_feat= self.feat(x, with_intermediate=True)
            return feat1, feat2, feat3, gl_feat
        else:
            gl_feat, trans, trans_feat = self.feat(x)
            feature_global= gl_feat
            class_output= self.classifier(gl_feat)
            hidden_output= self.projection(gl_feat)

            return feature_global, class_output, hidden_output

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

def test_detail():
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

def test():
    import sys
    import time
    from util import parameter_number
    
    device = torch.device('cuda:4')
    inputs = torch.zeros(32, 3008, 3).to(device)
    model = PointNetCls(3).to(device) # 수정 40 > 2
    start = time.time()
    output = model(inputs)

    print("Inference time: {}".format(time.time() - start))
    print("Parameter #: {}".format(parameter_number(model)))
    print("Inputs size: {}".format(inputs.size()))
    print("Output size: {}".format(output.size()))

if __name__ == '__main__':
    test()    
