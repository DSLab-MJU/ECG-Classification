import os
from torch import load
from torch.nn import DataParallel
from RPN import RPN
from data_ import *
from backbone_ import *
from backbone_ import mb


class rpn_initor():
    def __init__(self):

        self.lr1 = 0.15
        self.max_pre = 0
        self.max_acc = 0
        self.max_recall = 0
        self.batch = 240
        os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
        device_ids = [0, 1]

        self.features = mb().eval()

        path = os.path.join(os.getcwd(), "base_max.p")
        tmp = load(path)

        self.features.load_state_dict(tmp) 
        self.RPN = RPN()
        self.features = self.features.cuda()
        self.RPN = self.RPN.cuda()

        self.RPN.apply(weights_init)
        self.features = DataParallel(self.features, device_ids=device_ids)
        self.RPN = DataParallel(self.RPN, device_ids=device_ids)