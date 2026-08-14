import sys
import time
import warnings
import os

from sklearn.cluster import k_means
warnings.filterwarnings("ignore") 
sys.path.append('./')
import numpy as np
import torch
import torch.nn.functional as F
from utils import metric, initialization_utils
from SCUAPR import netw, CustomDataset, load_multimodal_data, Encoder, Decoder, band_encoder
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f'using {DEVICE}')

class RecordResult:
    def __init__(self):
        self.best_result_record = None
        self.metric_name = None

    def update(self, result: dict, metric: str):
        if self.best_result_record is None or result[metric] > self.best_result_record[metric]:
            self.best_result_record = result
            self.metric_name = metric

    def get_best(self):
        return self.best_result_record



#---------------------------------------------------#
def predict(anchor_graph, n_cluster):
    anchor_graph = torch.abs(anchor_graph)
    norm_anchor_graph = F.softmax(anchor_graph/5, dim=0)
    u, s, vh = torch.linalg.svd(norm_anchor_graph, full_matrices=False)  # # descending order
    # # top-k rows corresponding to singular values
    embedding = vh[:n_cluster, :].t().cpu().numpy()
    # # perform k-means to obtain labels
    cluster_centers, labels, _ = k_means(embedding, n_clusters=n_cluster, random_state=None)
    return labels


#===========Houston============================#
# the_dataset =  "Houston"
# dataset_root =  './datasets/Houston/Houston'
# model_path =   "save/Houston"
# n_anchor = 2
# nb_comps = 1
# LEARNING_RATE = 9
# epoch = 100
# seed =  43


#===========MUUFL============================#
# the_dataset =  "MUUFL"
# dataset_root =  './datasets/MUUFL/MUUFLGfport'
# model_path =   "save/MUUFL-Gulfport"
# n_anchor = 8
# nb_comps = 9
# LEARNING_RATE = 0.3
# epoch = 100
# seed =  111

#===========Trento============================#
the_dataset = "Trento"
dataset_root =  './datasets/Trento/'
model_path =   "save/Trento"
n_anchor = 3
nb_comps = 4
LEARNING_RATE = 4
epoch = 100
seed = 42  

is_labeled_pixel = False
verbose =  True
is_verify = False

root = dataset_root

initialization_utils.set_global_random_seed(seed=seed)
# prepare data
if the_dataset == "Houston":
    im_1, im_2 = 'HSI', 'Lidar'
    gt_ = 'GT'
    img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
    data_name = (im_1, im_2)
elif the_dataset == "Trento":
    im_1, im_2 = 'Trento-HSI', 'Trento-Lidar'
    gt_ = 'Trento-GT'
    img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
    # img_path = (root + im_2 + '.mat', )
    data_name = (im_1, im_2)
elif the_dataset == "MUUFL":
    im_1, im_2 = 'HSI', 'LiDAR_data_first_return' 
    gt_ = 'GT'
    img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
    data_name = (im_1, im_2)
else:
    raise NotImplementedError
gt_path = root + gt_ + '.mat'

for i, p_ in enumerate(img_path):
    print(f'modality #{i + 1}: {p_}')


x, gt = load_multimodal_data(gt_path, *img_path, is_labeled=is_labeled_pixel, nb_comps=nb_comps, device=DEVICE)
y = gt.reshape((-1,))
dataset = CustomDataset(x)

BATCH_SIZE = 1

dataloader_train = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)


if is_labeled_pixel:
    class_num = len(np.unique(y))
    y_labeled = np.copy(y)
else:
    indx_labeled = np.nonzero(y)[0]
    y_labeled = y[indx_labeled]
    class_num = len(np.unique(y)) - 1

data_size = x[0].shape[0] * x[0].shape[1]
spatial_size = gt.shape

print('# classes:', class_num)

anchor_num = class_num*n_anchor

start_time = time.time()

record = RecordResult()

encoder = Encoder(nb_comps)
decoder = Decoder(nb_comps)
band = band_encoder(x[1].shape[2], nb_comps)

model = netw(anchor_num, data_size, DEVICE, encoder, decoder, band).to(DEVICE)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

if is_verify:
    model.load_state_dict(torch.load('Houston.pth'))

if verbose:
    progress_bar = tqdm(total=epoch, desc="Training")

for e in range(epoch):
    
    if is_verify:
        with torch.no_grad():
            pass
    else:            
        for step, (x0, x1) in enumerate(dataloader_train):
            x0 = torch.tensor(x0).to(DEVICE)
            x1 = torch.tensor(x1).to(DEVICE)
            x0 = x0.permute(0, 3, 1, 2)
            x1 = x1.permute(0, 3, 1, 2)
            tl = [x0, x1]
            
            optimizer.zero_grad()
            A0, A1, x_hat0, x_hat1, h0, h1, x_1 = model(tl)

            
            tl = [x0, x_1]
            
            anchor =[A0, A1]
            x_hat = [x_hat0, x_hat1]
            h = [h0, h1]
            
            loss, loss_rec, loss_sub, loss_nor = model.loss_t(tl , anchor, x_hat, h)
            loss.backward()
            optimizer.step()

    if verbose:
        progress_bar.update(1)

    Z = model.Z.detach()
    y_pred = predict(Z, class_num)

    if not is_labeled_pixel:
        y_pred_labeled = y_pred[indx_labeled]
        y_pred_2D = y_pred.reshape(gt.shape)
    else:
        y_pred_labeled = y_pred
    acc, kappa, nmi, ari, pur, bcubed_F, ca= metric.cluster_accuracy(y_labeled, y_pred_labeled)
    if verbose:
        print('OA = {:.4f} Kappa = {:.4f} NMI = {:.4f} ARI = {:.4f} Purity = {:.4f}  BCubed F = {:.4f}'.format(acc, kappa,
                                                                                                            nmi, ari,
                                                                                                            pur,bcubed_F))

    new_results = {
        'OA':float(round(acc, 4)),
        'Kappa':float(round(kappa, 4)),
        'NMI':float(round(nmi, 4)),
        'ARI':float(round(ari, 4)),
        'Purity':float(round(pur, 4)),
        'BCubed F':float(round(bcubed_F, 4)),
        
    }                                                                                               
        
    new_results['Z'] = Z.cpu().numpy()
    new_results['y_pred_2D'] = y_pred_2D
    new_results['anchor_num'] = n_anchor
    new_results['epoch'] = e
    new_results['learning rate'] = LEARNING_RATE

    record.update(new_results, 'OA')
if verbose:    
    progress_bar.close()

max_oa_metrics = record.get_best()


print('------------------------------------------------------------------------------')
for key, value in max_oa_metrics.items():
    if key not in [
'y_pred_2D'
,'Z']:
        print(f'{key} : {value}')
print('------------------------------------------------------------------------------')



running_time = time.time() - start_time

print(f"运行时间：{running_time:.2f} 秒")