import sys
sys.path.append("../../")
from util import parameter_number
from sklearn.metrics import accuracy_score
import torch
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict
import time
import numpy as np
import os 
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from torch.utils.tensorboard import SummaryWriter

class Manager():
    def __init__(self, model, args):
        self.args_info = args.__str__()
        self.device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

        if args.load:
            model.load_state_dict(torch.load(args.load)['cls_net'])


        self.cls_net= model.to(self.device)
        self.cls_opt = optim.Adam(self.cls_net.parameters(), lr=args.lr)
        self.cls_criterion = nn.CrossEntropyLoss()

        self.epoch = args.epoch

        self.lr_scheduler = optim.lr_scheduler.StepLR(self.cls_opt, step_size= 10, gamma= 0.5)


        self.save = args.save
        self.record_interval = args.interval
        self.record_file = None
        if args.record:
            self.record_file = open(args.record, 'w')
        self.best = {"epoch": 0, "acc": 0}

    def record(self, info):
        print(info)
        if self.record_file:
            self.record_file.write(info + '\n')
        
    def train(self, train_data, test_data):
        self.record("*****************************************")
        self.record("Hyper-parameters: {}".format(self.args_info))
        self.record("Model parameter number: {}".format(parameter_number(self.cls_net)))
        self.record("Model structure:\n{}".format(self.cls_net.__str__()))
        self.record("*****************************************")
        
        for epoch in range(self.epoch):
            self.cls_net.train()
            train_loss = 0
            train_label = LabelContainer()

            for i, (points, gt) in enumerate(train_data):
                points = points.to(self.device)
                gt = gt.view(-1,).to(self.device)
                
                out = self.cls_net.forward_cls_feat(points)
                loss = self.cls_criterion(out, gt)
                
                self.cls_opt.zero_grad()
                loss.backward()
                self.cls_opt.step()
                train_loss += loss.item()
    
                pred = torch.max(out, 1)[1]
                train_label.add(gt, pred)
                

                if (i + 1) % self.record_interval == 0:
                    self.record(' epoch {:3d} step {:5d} | avg loss: {:.5f} | avg acc: {:.5f}'.format(epoch +1, i+1, train_loss/(i + 1), train_label.get_acc()))
            
            train_loss /= (i+1)
            train_acc = train_label.get_acc()
            test_loss, test_acc = self.test(test_data)
                        
            if test_acc > self.best['acc']:
                self.best['epoch'] = epoch + 1
                self.best['acc'] = test_acc
                if self.save:
                    torch.save({'cls_net':self.cls_net.state_dict()}, self.save)
            
            self.record('= Epoch {} | Tain Loss: {:.5f} Train Acc: {:.3f} | Test Loss: {:.5f} Test Acc: {:.3f} | Best Acc: {:.3f}\n'.format(epoch + 1, train_loss, train_acc, test_loss, test_acc, self.best['acc']))
            self.lr_scheduler.step()

        self.record('* Best result at {} epoch with test acc {}'.format(self.best['epoch'], self.best['acc']))
    

    def test(self, test_data):
        self.cls_net.eval()
        test_loss = 0
        test_label = LabelContainer()

        for i, (points, gt) in enumerate(test_data):
            points = points.to(self.device)
            gt = gt.view(-1,).to(self.device)
            out = self.cls_net.forward_cls_feat(points)
    
            loss = self.cls_criterion(out, gt)     
            test_loss += loss.item()
            pred = torch.max(out, 1)[1]
            test_label.add(gt, pred)

        test_loss /= (i+1)
        test_acc = test_label.get_acc()
        self.record(' Test Acc: {:.3f} \n'.format(test_acc))
        predictions = test_label.get_predictions()
        return test_loss, test_acc # predictions






class LabelContainer():
    def __init__(self):
        self.has_data = False
        self.gt = None
        self.pred = None
    
    def add(self, gt, pred):
        gt = gt.detach().cpu().view(-1)
        pred = pred.detach().cpu().view(-1)
        if self.has_data == False:
            self.has_data = True
            self.gt = gt
            self.pred = pred
        else:
            self.gt = torch.cat([self.gt, gt])
            self.pred = torch.cat([self.pred, pred])

    def get_acc(self):
        return accuracy_score(self.gt, self.pred)
    def get_predictions(self):
        return self.pred if self.has_data else None
    


        