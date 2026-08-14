
import numpy as np
import torch
from torch.utils.data import Dataset
from Preprocessing import Processor
from sklearn.decomposition import PCA



class band_encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(band_encoder, self).__init__()
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.rl = torch.nn.ReLU()
    def forward(self, x):
        n, b, h, w, = x.shape
        
        x = x.permute(0, 2, 3, 1)
        x = x.reshape((h*w, b))
        
        x = self.linear(x)
        x = self.bn(x)
        
        
        x = x.reshape(n, h, w, self.out_channels)
        x = x.permute(0, 3, 1, 2)
        return x
    
class Encoder(torch.nn.Module):
    def __init__(self, channels):
        super(Encoder, self).__init__()
     
        self.conv1 = torch.nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding='same')  
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same') 
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same') 
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same') 
        self.conv5 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same')
        
        
        self.bn1 = torch.nn.BatchNorm2d(16) 
        self.bn2 = torch.nn.BatchNorm2d(32) 
        self.bn3 = torch.nn.BatchNorm2d(64) 
        self.bn4 = torch.nn.BatchNorm2d(128) 
        self.bn5 = torch.nn.BatchNorm2d(128) 

        
        self.rl = torch.nn.ReLU()
#==========================================================================================#       


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.rl(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.rl(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.rl(x)
        
        return x

class Decoder(torch.nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.deconv1 = torch.nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)  
        self.deconv2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)   
        self.deconv3 = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1) 
        self.deconv4 = torch.nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1)   
        self.deconv5 = torch.nn.ConvTranspose2d(16, channels, kernel_size=3, stride=1, padding=1)   
        self.bn1 = torch.nn.BatchNorm2d(128)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.bn4 = torch.nn.BatchNorm2d(16)
        
        self.rl = torch.nn.ReLU()
        
        

    def forward(self, x):
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.rl(x)
        
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.rl(x)
        
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.rl(x)
        
        x = self.deconv4(x)
        x = self.bn4(x)
        x = self.rl(x)
        
        x = self.deconv5(x)

        return x





class netw(torch.nn.Module):
    def __init__(self, anchors_num, nsamples, device, encoder, decoder, band_encoder):
        super(netw, self).__init__()
        self.device = device
        
        self.anchor_num = anchors_num
        self.Z = torch.nn.Parameter(torch.ones((anchors_num, nsamples), device=self.device), requires_grad=True)
        self.mse = torch.nn.MSELoss() 
        self.rl = torch.nn.ReLU()
        self.proj = torch.randn((nsamples, anchors_num), device=self.device)
        self.bn = torch.nn.BatchNorm1d(anchors_num)

        
        self.encoder = encoder
        self.decoder = decoder
        self.band_encoder = band_encoder

        
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                
                if m.bias is not None:
                    m.bias.data.zero_()
                   
            elif isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

    
    def getanchor(self, x):
        n, b, h, w = x.shape
        
        x = x.permute(0, 2, 3, 1)
        x = x.reshape((h*w, b))
        x = x.t() @ self.proj
        x = self.bn(x)

        return x

    def forward(self, x):
        
        x_1 = self.band_encoder(x[1])
        h0 = self.encoder(x[0])
        
        A0 = self.getanchor(h0)
        x_hat0 = self.decoder(h0)
        
        h1 = self.encoder(x_1)
        
        A1 = self.getanchor(h1)
        x_hat1 = self.decoder(h1)
        
        
        
        return A0, A1, x_hat0, x_hat1, h0, h1, x_1
       
    
    def loss_t(self, x, anchor, x_hat, h_):
        
        A0 = anchor[0]
        A1 = anchor[1]
        
        
        
        
        for i in range(len(x)):
            n, b, h, w = x[i].shape
            x[i] = x[i].permute(0, 2, 3, 1)
            x[i] = x[i].reshape((h*w, b))
            
        for i in range(len(x_hat)):
            n, b, h, w = x_hat[i].shape
            x_hat[i] = x_hat[i].permute(0, 2, 3, 1)
            x_hat[i] = x_hat[i].reshape((h*w, b))
            
        for i in range(len(h_)):
            n, b, h, w = h_[i].shape
            h_[i] = h_[i].permute(0, 2, 3, 1)
            h_[i] = h_[i].reshape((h*w, b))
        
        
        
        s1 = self.mse(x[0], x_hat[0]) + self.mse(x[1], x_hat[1])
        s2 = (
            self.mse(h_[0].t(), torch.matmul(A0, self.Z)) + 
            self.mse(h_[1].t(), torch.matmul(A1, self.Z))
            )
        
        s3 = torch.norm(self.Z , p=2)
        

        
        s =  s1 + s2 + s3

        return s, s1, s2, s3
    
    
    


class CustomDataset(Dataset):
    def __init__(self, data_list):
        
        self.data1 = data_list[0].reshape((1, data_list[0].shape[0], data_list[0].shape[1], data_list[0].shape[2]))
        self.data2 = data_list[1].reshape((1, data_list[1].shape[0], data_list[1].shape[1], data_list[1].shape[2]))
        self.length = self.data1.shape[0]
  
    def __len__(self):
       
        
        return self.length

    def __getitem__(self, idx):
        
        sample1 = self.data1[idx]
        sample2 = self.data2[idx]
        
        
        return (torch.tensor(sample1, dtype=torch.float32),
                torch.tensor(sample2, dtype=torch.float32))
        


def load_multimodal_data(gt_path, *src_path, is_labeled=True, nb_comps, device):
    p = Processor()
    n_modality = len(src_path)
    modality_list = []

    
    img, gt = p.prepare_data(src_path[1], gt_path)
    nb = img.shape[2]
    
    for i in range(n_modality):
        img, gt = p.prepare_data(src_path[i], gt_path)

        n_row, n_column, n_band = img.shape
        
        modality_list.append(img)
        
    n_row, n_column, n_band = modality_list[0].shape    
    pca = PCA(n_components=nb_comps)
    modality_list[0] = pca.fit_transform(modality_list[0].reshape(n_row*n_column, n_band)).reshape((n_row, n_column, nb_comps))
    print('pca shape: %s, percentage: %s' % (modality_list[0].shape, np.sum(pca.explained_variance_ratio_)))    

  
    return modality_list, gt