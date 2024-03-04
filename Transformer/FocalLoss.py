import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma 
        self.average = average 

        if alpha is None: 
            self.alpha = torch.ones(class_num, 1) 
        else: 
            if isinstance(alpha, torch.Tensor): 
                self.alpha = alpha
            else:
                self.alpha = torch.Tensor(alpha)

    def forward(self, inputs, targets, device):
        N, C = inputs.size() 
        P = F.softmax(inputs, dim=1) 

        class_mask = inputs.new_zeros(N, C)
        ids = targets.view(-1, 1) 
        class_mask.scatter_(1, ids, 1.) 
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(device)
        alpha = self.alpha[ids.view(-1)] 

        probs = (P * class_mask).sum(1).view(-1, 1) 

        log_p = probs.log() 

        batch_loss = -alpha * torch.pow((1 - probs),self.gamma) * log_p 

        if self.average: 
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss