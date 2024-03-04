import random
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import DataParallel
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from fastdtw import fastdtw
import pywt
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

# Adapted from: https://github.com/TeaPearce/Conditional_Diffusion_MNIST
ent_num = '02'
seed = 42
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def norm_data2(data):
    df = pd.DataFrame(data)
    df = df.transpose()
    norm = minmax_scale(df)
    norm_data = norm.transpose()
    return norm_data

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

class ResidualConvBlock(nn.Module): 
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False) -> None:
        super().__init__()
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.GELU(),)
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.GELU(),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels: 
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414 
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)] 
        self.model = nn.Sequential(*layers)

    def forward(self, x): 
        return self.model(x)


class UnetUp(nn.Module): 
    def __init__(self, in_channels, out_channels, odd):
        super(UnetUp, self).__init__()
        if odd == 0: 
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
                      ResidualConvBlock(out_channels, out_channels),
                      ResidualConvBlock(out_channels, out_channels),]
        else: 
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 2, 2, output_padding = 1),
                      ResidualConvBlock(out_channels, out_channels),
                      ResidualConvBlock(out_channels, out_channels),]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1) 
        x = self.model(x) 
        return x


class EmbedFC(nn.Module): 
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        layers = [nn.Linear(input_dim, emb_dim), 
                  nn.GELU(),
                  nn.Linear(emb_dim, emb_dim),]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim) 
        return self.model(x) 


class ContextUnet(nn.Module): 
    def __init__(self, in_channels, n_feat = 256, n_classes=3): 
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True) 

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(5), nn.GELU()) 

        self.timeembed1 = EmbedFC(1, 2*n_feat) 
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat) 
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 5, 5), 
                                 nn.GroupNorm(8, 2 * n_feat),
                                 nn.ReLU(),)

        self.up1 = UnetUp(4 * n_feat, n_feat, 1) 
        self.up2 = UnetUp(2 * n_feat, n_feat, 0)
        self.out = nn.Sequential(nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
                                 nn.GroupNorm(8, n_feat),
                                 nn.ReLU(),
                                 nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),)

    def forward(self, x, c, t, context_mask): 
        x = self.init_conv(x) 
        down1 = self.down1(x) 
        down2 = self.down2(down1) 
        hiddenvec = self.to_vec(down2) 
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float) 

        context_mask = context_mask[:, None] 
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask)) 
        c = c * context_mask
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1) 
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1) 
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec) 
        up2 = self.up1(cemb1*up1+ temb1, down2)  
        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1)) 
        return out


def ddpm_schedules(beta1, beta2, T): 
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)" 

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1 
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp() 

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {"alpha_t": alpha_t, 
            "oneover_sqrta": oneover_sqrta,  
            "sqrt_beta_t": sqrt_beta_t,  
            "alphabar_t": alphabar_t,  
            "sqrtab": sqrtab,  
            "sqrtmab": sqrtmab, 
            "mab_over_sqrtmab": mab_over_sqrtmab_inv, 
            }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T 
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c): 
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  
        noise = torch.randn_like(x)  

        x_t = (self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise)

        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device) 
        m = self.nn_model(x_t, c, _ts / self.n_T, context_mask)
        out = self.loss_mse(noise, m) 
        return out, noise, m

def MMD(x, y, kernel, device):
    x = x.to(device)
    y = y.to(device)
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx 
    dyy = ry.t() + ry - 2. * yy 
    dxy = rx.t() + ry - 2. * zz 
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
           
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
    return torch.mean(XX + YY - 2. * XY)


def val(val_dataloader, model, device, test):
    val_pbar = tqdm(val_dataloader)
    loss_ema = None
    tot_loss = []
    all_ori_s, all_re_s = np.empty((0,484), int), np.empty((0,484), int)
    all_ori_v, all_re_v = np.empty((0,484), int), np.empty((0,484), int)
    all_ori_f, all_re_f = np.empty((0,484), int), np.empty((0,484), int)
    model.eval()
    with torch.no_grad():
        for x in val_pbar: 
            x, c = torch.split(x, [484, 1], dim=1)
            x = x.view(-1, 1, 22, 22)
            x = x.float()
            c = c.view(-1)
            x = x.to(device)
            c = c.to(device)
            c = c.type(torch.int64)
            loss, noise, re = model(x, c)
            loss = loss.mean()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item() 
            val_pbar.set_description(f"val_loss: {loss_ema:.4f}")
            tot_loss.append(loss_ema)
            if test == 1:
                for jj in range(len(noise)):
                    if c[jj] == 0:
                        all_ori_s = np.append(all_ori_s, noise[jj].view(-1, 484).detach().cpu().numpy(), axis = 0)
                        all_re_s = np.append(all_re_s, re[jj].view(-1, 484).detach().cpu().numpy(), axis = 0)
                    elif c[jj] == 1:
                        all_ori_v = np.append(all_ori_v, noise[jj].view(-1, 484).detach().cpu().numpy(), axis = 0)
                        all_re_v = np.append(all_re_v, re[jj].view(-1, 484).detach().cpu().numpy(), axis = 0)
                    elif c[jj] == 2:
                        all_ori_f = np.append(all_ori_f, noise[jj].view(-1, 484).detach().cpu().numpy(), axis = 0)
                        all_re_f = np.append(all_re_f, re[jj].view(-1, 484).detach().cpu().numpy(), axis = 0)
        
    if test == 1:
        all_dis_s, all_dis_v, all_dis_f = [], [], []
        all_cos_s, all_cos_v, all_cos_f = [], [], []
        mmd_s = MMD(torch.Tensor(all_ori_s), torch.Tensor(all_re_s), "rbf", device)
        mmd_v = MMD(torch.Tensor(all_ori_v), torch.Tensor(all_re_v), "rbf", device)
        mmd_f = MMD(torch.Tensor(all_ori_f), torch.Tensor(all_re_f), "rbf", device)

        for kk in range(len(all_ori_s)):
            ori = all_ori_s[kk].reshape(-1, 1)
            re = all_re_s[kk].reshape(-1, 1)
            distance, _ = fastdtw(ori, re, dist=euclidean)
            cos =  cos_sim(all_ori_s[kk], all_re_s[kk])
            all_dis_s.append(distance)
            all_cos_s.append(cos)
            
        for kk in range(len(all_ori_v)):
            ori = all_ori_v[kk].reshape(-1, 1)
            re = all_re_v[kk].reshape(-1, 1)
            distance, _ = fastdtw(ori, re, dist=euclidean)
            cos =  cos_sim(all_ori_v[kk], all_re_v[kk])
            all_dis_v.append(distance)
            all_cos_v.append(cos)

        for kk in range(len(all_ori_f)):
            ori = all_ori_f[kk].reshape(-1, 1)
            re = all_re_f[kk].reshape(-1, 1)
            distance, _ = fastdtw(ori, re, dist=euclidean)
            cos =  cos_sim(all_ori_f[kk], all_re_f[kk])
            all_dis_f.append(distance)
            all_cos_f.append(cos)

        return sum(tot_loss) / len(x)
    else:
        return sum(tot_loss) / len(x)

def WT(df, wavelet='db5', thresh=0.1):
    signal = df
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal

def train_ecg():
    n_epoch = 10000
    batch_size = 256
    n_T = 500 
    cuda = True
    device = torch.device("cuda" if cuda else "cpu")
    n_classes = 3 
    n_feat = 128 
    lrate = 1e-4
    save_dir = './' + ent_num + '/diff/'

    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1).cuda()
    ddpm = DataParallel(ddpm, device_ids=[0, 1])

    data = np.load('./' + ent_num + '/data/train_data.npy')
    label = np.load('./' + ent_num + '/data/train_label.npy')
    aa = pd.DataFrame(label)

    d1 = []
    max_cnt = aa.value_counts().values[0]
    a = aa.value_counts().values
    b = aa.value_counts().index
    for i in range(len(a)):
        if a[i] == a[0]:
            continue
        else:
            d1.append([b[i][0], (max_cnt-a[i])//10]) 
    
    norm_data = norm_data2(data)
    data_df = pd.DataFrame(norm_data) 
    data_df['label'] = label
    np.save(save_dir + 'ori_data',norm_data)
    np.save(save_dir + 'ori_label', label)

    del_0_idx = np.where(label==0)
    norm_data = np.delete(norm_data, del_0_idx, axis = 0)
    norm_data = norm_data[:, 8:-8]
    norm_label = np.delete(label, del_0_idx, axis = 0)
    norm_label = norm_label-1 

    tot_data = np.column_stack((norm_data, norm_label))

    train_d, test_d = train_test_split(tot_data, test_size = 0.2, random_state=seed)
    val_d, test_d = train_test_split(test_d, test_size = 0.5, random_state=seed)

    set_seed(seed)
    dataloader = torch.utils.data.DataLoader(train_d, batch_size, shuffle=True, num_workers=5)
    set_seed(seed)
    val_dataloader = torch.utils.data.DataLoader(val_d, batch_size, shuffle=True, num_workers=5)
    set_seed(seed)
    test_dataloader = torch.utils.data.DataLoader(test_d, batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    
    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x in pbar:
            optim.zero_grad()
            x, c = torch.split(x, [484, 1], dim=1)
            x = x.view(-1, 1, 22, 22)
            x = x.float()
            c = c.view(-1)
            x = x.to(device)
            c = c.to(device)
            c = c.type(torch.int64)
            loss, noise, re = ddpm(x, c)
            loss = loss.mean()
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item() 
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        val_loss = val(val_dataloader, ddpm, device, 0)

        if ep == 0:
            min_val_loss = 10000

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(ddpm.state_dict(), save_dir + "ddpm.pth")
            test_loss = val(test_dataloader, ddpm, device, 1)

if __name__ == "__main__":
    train_ecg()