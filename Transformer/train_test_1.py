import sys
import time
import sys
import os
import random
import numpy as np
from torch import save
from torch.optim import Adadelta
from torch.utils.data import DataLoader
import torch
from data_loader import loader

from detection_collate_mit import call_back
from rpn_tool_d import rpn_tool_d
from backbone_ import adjust_learning_rate
from eval_1sg import rpn_evalor

# Adapted from: https://github.com/QiXi9409/Simultaneous_ECG_Heartbeat
sys.path.append(os.getcwd())
eps = 10000
seed = 42
ent_num = '02'
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class model(rpn_evalor):

    def train_stage_1(self):
        time_start = time.time()
        set_seed(seed)
        data_set = loader(os.path.join(os.getcwd(), 'data_2'), {"seed": seed, "mode": "training"}, sp=1)
        set_seed(seed)
        data_set_test = loader(os.path.join(os.getcwd(), 'data_2'), {"seed": seed, "mode": "test"}, sp=1)
        set_seed(seed)
        data_set_eval = loader(os.path.join(os.getcwd(), 'data_2'), {"seed": seed, "mode": "eval"}, sp=1)

        set_seed(seed)
        data_loader = DataLoader(data_set, self.batch, True, collate_fn=call_back.detection_collate_RPN, num_workers=0)
        set_seed(seed)
        data_loader_test = DataLoader(data_set_test, self.batch, False, collate_fn=call_back.detection_collate_RPN,
                                      num_workers=0)
        set_seed(seed)
        data_loader_eval = DataLoader(data_set_eval, self.batch, False, collate_fn=call_back.detection_collate_RPN,
                                      num_workers=0)
        
        optim = Adadelta(self.RPN.parameters(), lr=self.lr1, weight_decay=1e-5)

        tool = rpn_tool_d()
        start_time = time.time()
        min_eval_loss = 1000000000
        for epoch in range(eps):  
            runing_losss = 0.0
            cls_loss = 0
            coor_loss = 0

            for data in data_loader:
                y = data[1]
                x = data[0].cuda()

                optim.zero_grad()
                with torch.no_grad():
                    x1, x2, x3, x4 = self.features(x)
                predict_confidence, box_predict = self.RPN(x1, x2, x3, x4)
                cross_entropy, loss_box = tool.get_proposal(predict_confidence, box_predict, y)
                loss_total = cross_entropy + loss_box
                loss_total.backward()
                optim.step()
                runing_losss += loss_total.item()
                cls_loss += cross_entropy.item()
                coor_loss += loss_box.item()
            end_time = time.time()
            print("epoch:{a}: loss:{b:.4f} spend_time:{c:.4f} cls:{d:.4f} cor{e:.4f} date:{ff}".format(a=epoch,
                                                                                                       b=runing_losss,
                                                                                                       c=int(
                                                                                                           end_time - start_time),
                                                                                                       d=cls_loss,
                                                                                                       e=coor_loss,
                                                                                                       ff=time.asctime()))
            start_time = end_time

            self.features.eval()
            self.RPN.eval()
            eval_loss = self.evalu(self.features, self.RPN, tool, data_loader_eval)
            if eval_loss < min_eval_loss:
                save(self.RPN.module.state_dict(),
                    os.path.join(os.getcwd(), 'rpn_a1_max.p'))
                save(self.features.module.state_dict(),
                    os.path.join(os.getcwd(), 'base_a1_max.p'))
                min_eval_loss = eval_loss

                self.RPN_eval(data_loader_test, {"epoch": epoch})

            self.features.train()
            self.RPN.train()
            tool.train_mode = True
            if epoch % 10 == 0 and epoch > 0:
                adjust_learning_rate(optim, 0.9, epoch, 50, 0.3)

    
    def evalu(self, features, RPN, tool, data_loader_eval):
        features.eval()
        RPN.eval()
        tool.train_mode = False
        runing_losss = 0.0
        with torch.no_grad():
            for data in data_loader_eval:
                y = data[1]
                x = data[0].cuda()

                x1, x2, x3, x4 = features(x)
                predict_confidence, box_predict = RPN(x1, x2, x3, x4)
                cross_entropy, loss_box = tool.get_proposal(predict_confidence, box_predict, y)
                loss_total = cross_entropy + loss_box

                runing_losss += loss_total.item()

        return runing_losss
    

if __name__ == '__main__':
    a = model()
    a.train_stage_1()