from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import DataParallel
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
import random
import time
import pywt

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

        out = self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask)) 
        return out

    def sample(self, n_sample, x_i, size, device, guide_w = 0.0, gen_label = []): 
        x_i = x_i.to(device)
        cc_i = torch.arange(0,4).to(device) 
        if len(gen_label) == 0:
            cc_i = cc_i.repeat(int(n_sample/cc_i.shape[0])) 
        else:
            cc_i = torch.tensor([], dtype=torch.int64)
            for label, count in gen_label:
                label_tensor = torch.full((count,), label, dtype=torch.int64) 
                cc_i = torch.cat((cc_i, label_tensor))

        cc_i = cc_i.to(device)
        context_mask = torch.zeros_like(cc_i).to(device) 

        c_i = cc_i.repeat(2) 
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. 
        x_i_store = [] 
        for i in range(self.n_T, 0, -1): 
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1) 

            x_i = x_i.repeat(2,1,1,1) 
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0 

            eps = self.nn_model(x_i, c_i, t_is, context_mask) 
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2 
            x_i = x_i[:n_sample] 
            x_i = (self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z) 

        return x_i, x_i_store, cc_i

def val(val_dataloader, model, device):
    val_pbar = tqdm(val_dataloader)
    loss_ema = None
    tot_loss = []
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
            loss, re = model(x, c)
            loss = loss.mean()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item() 
            val_pbar.set_description(f"val_loss: {loss_ema:.4f}")
            tot_loss.append(loss_ema)

    return sum(tot_loss) / len(x)

def WT(df, wavelet='db5', thresh=0.1):
    signal = df
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal

def train_ecg():
    n_T = 500 
    cuda = True
    device = torch.device("cuda" if cuda else "cpu")
    n_classes = 4
    n_feat = 128 
    save_dir = './' + ent_num + '/diff2/' 
    ws_test = [2.0] 

    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1).cuda()
    ddpm = DataParallel(ddpm, device_ids=[0, 1])

    data = np.load('./final_train_data.npy')
    data = np.pad(data, ((0, 0), (2, 2)), 'constant', constant_values=0)
    data[:,:2] = data[:, 2:3]
    data[:, -2:] = data[:, -3:-2]
    label = np.load('./final_train_label.npy')
    aa = pd.DataFrame(label)
    ccnt = 200
    d1, d2 = [], []
    max_cnt = aa.value_counts().values[0]
    a = aa.value_counts().values
    b = aa.value_counts().index
    for i in range(len(a)):
        if a[i] == a[0]:
            continue
        else:
            if b[i][0] == 0: 
                d1.append([3, (max_cnt-a[i])]) 
                d2.append([3, (max_cnt-a[i])//ccnt])
            elif b[i][0] == 2:  
                d1.append([0, (max_cnt-a[i])]) 
                d2.append([0, (max_cnt-a[i])//ccnt])
            elif b[i][0] == 3: 
                d1.append([1, (max_cnt-a[i])]) 
                d2.append([1, (max_cnt-a[i])//ccnt])
            elif b[i][0] == 4: 
                d1.append([2, (max_cnt-a[i])]) 
                d2.append([2, (max_cnt-a[i])//ccnt])
    
    norm_data = norm_data2(data)
    data_df = pd.DataFrame(norm_data) 
    data_df['label'] = label
    np.save(save_dir + 'ori_data',norm_data)
    np.save(save_dir + 'ori_label', label)

    del_0_idx = np.where(label==1) 
    norm_data = np.delete(norm_data, del_0_idx, axis = 0)
    norm_label = np.delete(label, del_0_idx, axis = 0)
    norm_label = norm_label-2 
    norm_label = np.where(norm_label == -2, 3, norm_label)
    aa = pd.DataFrame(norm_label)
    
    n_sample = sum([item[1] for item in d1])
    x_i = torch.randn(n_sample, *(1, 22, 22)).numpy()
    np.save(save_dir + 'x_i', x_i)

    x_i = np.load(save_dir + 'x_i.npy')
    x_i = torch.Tensor(x_i)

    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1).cuda()
    ddpm = nn.DataParallel(ddpm, device_ids=[0, 1])
    ddpm.load_state_dict(torch.load(save_dir + 'ddpm.pth'))

    ddpm.eval()
    ccnt_cnt = 0
    for i in d2:
        ccnt_cnt += i[1]
    for k in range(ccnt):
        with torch.no_grad():
            xx_i = x_i[k*ccnt_cnt:(k+1)*ccnt_cnt]
            xx_i = xx_i.to(device)

            x_gen, _, y_gen = ddpm.module.sample(len(xx_i), xx_i, (1, 22, 22), device, guide_w=ws_test[0], gen_label=d2) 
            x_gen = x_gen.detach().cpu().numpy()
            x_gen = x_gen.reshape(-1, 484)

            idx0 = torch.nonzero(y_gen == 0).flatten()[:4]
            idx1 = torch.nonzero(y_gen == 1).flatten()[:4]
            idx2 = torch.nonzero(y_gen == 2).flatten()[:4]
            idx3 = torch.nonzero(y_gen == 3).flatten()[:4]
            combined_indices = []
            for i in range(4):
                combined_indices.extend([idx0[i], idx1[i], idx2[i], idx3[i]])

            for gi in range(len(x_gen)):
                x_gen[gi] = WT(x_gen[gi].flatten())
            
            x_gen = x_gen[:, 2:-2]
            y_gen += 2
            y_gen = y_gen.detach().cpu().numpy()
            y_gen = np.where(y_gen == 5, 0, y_gen)
            aa = pd.DataFrame(y_gen)
            np.save(save_dir + 'diff_data_' + str(k), x_gen) 
            np.save(save_dir + 'diff_label_' + str(k), y_gen)

if __name__ == "__main__":
    train_ecg()