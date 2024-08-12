import sys
from dataset_modelnet import ModelNet_pointcloud
from util import parameter_number
from sklearn.metrics import accuracy_score
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import time
import numpy as np
from generator import Autoencoder

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

from torch.utils.tensorboard import SummaryWriter

from con_losses import SupConLoss


class Manager():
    def __init__(self, model, args):
        self.args_info = args.__str__()
        self.device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
         
        
        self.lr= args.lr
        self.support_num= args.support
        self.neighbor_num= args.neighbor


        self.w_cls= args.w_cls
        self.w_cyc= args.w_cyc
        self.w_info= args.w_info
        self.w_div= args.w_div
        self.div_thresh= args.div_thresh
        self.w_tgt= args.w_tgt
        self.gen_name= args.gen_name
        self.n_tgt= args.n_tgt
        
        self.tgt_epochs= args.tgt_epochs
        self.tgt_epochs_fixg= args.tgt_epochs_fixg

        
        
        model.load_state_dict(torch.load(args.load)['cls_net'])
        
        self.saved_ep= 0
        self.saved_tgt_domain=0

        #if model.load_state_dict(torch.load(args.load)['epoch'])>0 is not None and model.load_state_dict(torch.load(args.load)['epoch'])>0:
        #    self.saved_ep= model.load_state_dict(torch.load(args.load)['epoch'])
        #if model.load_state_dict(torch.load(args.load)['epoch'])>0 is not None and model.load_state_dict(torch.load(args.load)['epoch'])>0:
        #    self.saved_tgt_domain= model.load_state_dict(torch.load(args.load)['target'])
        
        self.src_net= model.to(self.device)
        self.src_opt = optim.Adam(self.src_net.parameters(), lr=args.lr)

        self.cls_criterion = nn.CrossEntropyLoss()
        self.con_criterion= SupConLoss(device= self.device).to(self.device)
        

        self.g1_list=[]
        

        if self.gen_name=='adain':
            self.g1_net= Autoencoder(model, self.support_num, self.neighbor_num, args)
            self.g2_net= Autoencoder(model, self.support_num, self.neighbor_num, args)
            
            self.g1_net= self.g1_net.to(self.device)
            self.g2_net= self.g2_net.to(self.device)
            
            self.g1_opt= optim.Adam(self.g1_net.parameters(), lr=self.lr)
            self.g2_opt= optim.Adam(self.g2_net.parameters(), lr= self.lr)
            

        self.save = args.save
        self.save_img= args.save_img
       
        if self.save_img and args.mode== 'train':
            self.img_root= os.path.join(self.save, 'gen_img')
            if not os.path.exists(self.img_root):
                os.makedirs(self.img_root)
        
        if self.save:
            self.g1_root = os.path.join(self.save, 'g1')
            if not os.path.exists(self.g1_root) and args.mode== 'train':
                os.makedirs(self.g1_root)

        
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
        self.record("Model parameter number: {}".format(parameter_number(self.src_net)))
        self.record("Model structure:\n{}".format(self.src_net.__str__()))
        self.record("*****************************************")


        for i_tgt in range(self.saved_tgt_domain, self.n_tgt):
            i_tgt= i_tgt
            print(f'target domain {i_tgt}')
            self.best['epoch']= self.best['acc']= 0
            #g1_net, g2_net, g1_opt, g2_opt= self.g1_net, self.g2_net, self.g1_opt, self.g2_opt
            for ep in range(self.saved_ep, self.tgt_epochs):
                t1= time.time()
                flag_fixG= False # G is locked

                if (self.tgt_epochs_fixg is not None) and (ep >= self.tgt_epochs_fixg):
                    flag_fixG = True

                train_label = LabelContainer()
                loss_list= []
                time_list= []

                self.src_net.eval()


                for i, (points, gt) in enumerate(train_data):
                    points = points.to(self.device)
                    gt = gt.view(-1,).to(self.device)
                    
                    if len(self.g1_list)>0: # If there are generators, I will choice one of them as source data
                        idx = np.random.randint(0, len(self.g1_list))

                        if self.gen_name == 'adain':
                            with torch.no_grad():
                                points_src2 = self.g1_list[idx](points)

                            # mix term 추가해야함.
                            #points_src3= ~~
                    # New point generation
                    if self.gen_name == 'adain':
                        points_tgt = self.g1_net(points)
                        points_tgt2 = self.g1_net(points)
                    
                    _, class_src1, hidden_src1= self.src_net(points)

                    if len(self.g1_list)>0: # feedforward
                        _, class_src2, hidden_src2 = self.src_net(points_src2)
                        #_, class_src3, hidden_src3= self.src_nut(points_src3)

                        hidden_src= torch.cat([hidden_src1.unsqueeze(1), hidden_src2.unsqueeze(1)], dim=1)
                        src_loss= self.cls_criterion(class_src1, gt) + self.cls_criterion(class_src2, gt) # +self.cls_criterion(class_src3, gt)
        
                    else:
                        hidden_src = hidden_src1.unsqueeze(1)
                        src_loss= self.cls_criterion(class_src1, gt)

                    _, class_tgt1, hidden_tgt1 = self.src_net(points_tgt)
                    
                    tgt_cls_loss = self.cls_criterion(class_tgt1, gt)
                    
                    #breakpoint()     
                    hidden_all= torch.cat([hidden_tgt1.unsqueeze(1), hidden_src], dim=1)           
                    con_loss= self.con_criterion(hidden_all, adv=False)                  
                    loss= src_loss+ self.w_tgt*con_loss +self.w_tgt*tgt_cls_loss # tgt_cls_loss 삭제할 지 말 지 생각해보기
                    
                    src_net= self.src_net

                    self.src_opt.zero_grad()


                    if flag_fixG:
                        loss.backward()
                    else:
                        loss.backward(retain_graph=True)
                        
                    self.src_opt.step()

                    if flag_fixG:
                        con_loss_adv= torch.tensor(0)
                        div_loss= torch.tensor(0)
                        cyc_loss= torch.tensor(0)
                    else:
                        _, class_tgt1, hidden_tgt1 = src_net(points_tgt)
                        tgt_cls_loss= self.cls_criterion(class_tgt1, gt)

                        idx = np.random.randint(0, hidden_src.size(1))

                        hidden_all = torch.cat([hidden_tgt1.unsqueeze(1), hidden_src[:,idx:idx+1].detach()], dim=1)

                        con_loss_adv = self.con_criterion(hidden_all, adv=True)
                        
                        points_tgt_rec= self.g2_net(points_tgt)
                        cyc_loss= F.mse_loss(points_tgt_rec, points)
                        
                        if self.gen_name== 'adain':
                            div_loss= (points_tgt-points_tgt2).abs().mean([1]).clamp(max=self.div_thresh).mean()

                        loss= self.w_cls*tgt_cls_loss -self.w_div*div_loss + self.w_cyc*cyc_loss + self.w_info*con_loss_adv

                        self.g1_opt.zero_grad()
                        if self.g2_opt is not None:
                            self.g2_opt.zero_grad()
                            
                        loss.backward(retain_graph=True)
                        self.g1_opt.step()
                        if self.g2_opt is not None:
                            self.g2_opt.step()


                    loss_list.append([src_loss.item(), tgt_cls_loss.item(), con_loss.item(), con_loss_adv.item(), div_loss.item(), cyc_loss.item()])

                    pred = torch.max(class_src1, 1)[1]
                    train_label.add(gt, pred)

                    if (i + 1) % self.record_interval == 0:
                        self.record(' epoch {:3d} step {:5d} | avg acc: {:.5f}'.format(ep +1, i+1, train_label.get_acc()))
                        self.record('= src_loss {:.5f} | tgt_cls_loss: {:.5f} con_loss: {:.5f} | con_loss_adv: {:.5f} div_loss: {:.5f} | cyc_loss: {:.5f}\n'.format(src_loss, tgt_cls_loss, con_loss, con_loss_adv, div_loss, cyc_loss))

                src_loss, tgt_cls_loss, con_loss, con_loss_adv, div_loss, cyc_loss= np.mean(loss_list, axis= 0)

                self.src_net.eval()

                train_acc = train_label.get_acc()
                test_loss, test_acc = self.test(test_data)
                            
                if test_acc > self.best['acc']:
                    self.best['epoch'] = i_tgt*self.tgt_epochs + ep + 1
                    self.best['acc'] = test_acc
                    if self.save:
                        torch.save({'cls_net':self.src_net.state_dict(), 'target': i_tgt, 'epoch': ep}, os.path.join(self.save, f'{i_tgt}_best.pkl'))
                
                t2= time.time()

                self.record('= Epoch {} | Train Acc: {:.3f} | Test Acc: {:.3f} | Best Acc: {:.3f}\n'.format(i_tgt*self.tgt_epochs + ep + 1, train_acc, test_acc, self.best['acc']))
                self.record('= src_loss {:.5f} | tgt_cls_loss: {:.5f} con_loss: {:.5f} | con_loss_adv: {:.5f} div_loss: {:.5f} | cyc_loss: {:.5f}\n'.format(src_loss, tgt_cls_loss, con_loss, con_loss_adv, div_loss, cyc_loss))
                
                g1_all= self.g1_list + [self.g1_net]
                """
                if self.save_img:
                    with torch.no_grad():
                        for j in range(len(g1_all)):
                            selected_points= g1_all[j](points[:2], rand=True)

                            fig = plt.figure(figsize=(10, 7))

                            full_path= os.path.join(self.img_root, f'generated_image{i_tgt*self.tgt_epochs+ep+1}.png')

                            ax = fig.add_subplot(111, projection='3d')
                            ax.scatter(selected_points[:, 0],  # X 좌표
                                        selected_points[:, 1],  # Y 좌표
                                        selected_points[:, 2],  # Z 좌표
                                        c='blue',  # 포인트의 색상
                                        marker='o')  # 포인트의 모양
                            ax.set_xlabel('X axis')
                            ax.set_ylabel('Y axis')
                            ax.set_zlabel('Z axis')
                            plt.savefig(full_path)
                """

            torch.save({'g1':self.g1_net.state_dict(), 'target': i_tgt, 'epoch': ep}, os.path.join(self.g1_root, f'{i_tgt}.pkl'))            
            self.g1_list.append(self.g1_net)

    def test(self, test_data):
        
        self.src_net.eval()
        test_loss = 0
        test_label = LabelContainer()

        for i, (points, gt) in enumerate(test_data):
            points = points.to(self.device)
            gt = gt.view(-1,).to(self.device)
            _, out, _ = self.src_net(points)

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
    


        