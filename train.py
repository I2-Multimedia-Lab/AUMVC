import torch
from network import Network
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data

from metric import valid

import time
Dataname ='WIKI'
T = 1

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_iterations", default=200)
parser.add_argument("--con_iterations", default=200)
parser.add_argument("--tune_iterations", default=50)
parser.add_argument("--feature_dim", default=256)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument('--miss_rate', type=str, default=0.5)
parser.add_argument('--noise_rate', type=str, default=0.5)
parser.add_argument('--Gaussian_noise', type=str, default=0.4)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == "VOC":
    args.mse_iterations = 200
    args.miss_rate = 0.5
    args.noise_rate = 0.5
    args.aussian_noise = 0.4
    seed = 16
if args.dataset == "100Leaves":
    args.mse_iterations = 200
    args.miss_rate = 0.5
    args.noise_rate = 0.5
    args.aussian_noise = 0.4
    seed = 12
if args.dataset == "Cifar100":
    args.mse_iterations = 1000
    args.miss_rate = 0.5
    args.noise_rate = 0.5
    args.aussian_noise = 0.4
    seed = 1
if args.dataset == "WIKI":
    args.mse_iterations = 100
    args.miss_rate = 0.5
    args.noise_rate = 0.5
    args.aussian_noise = 0.4
    seed = 18
if args.dataset == "CUB":
    args.mse_iterations = 200
    args.miss_rate = 0.5
    args.noise_rate = 0.5
    args.aussian_noise = 0.4
    seed = 11
if args.dataset == "scene-15":
    args.mse_iterations = 200
    args.miss_rate = 0.5
    args.noise_rate = 0.5
    args.aussian_noise = 0.4
    seed = 21
if args.dataset == "Deep Animal":
    args.mse_iterations = 200
    args.miss_rate = 0.5
    args.noise_rate = 0.5
    args.aussian_noise = 0.4
    seed = 10

if args.dataset == "LandUse-21":
    args.mse_iterations = 200
    args.miss_rate = 0.5
    args.noise_rate = 0.5
    args.aussian_noise = 0.4
    seed = 1


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def mask(rows, cols, p):
    tensor = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        if i < int(rows * p):
            while True:
                row = np.random.randint(0, 2, size=cols)
                if np.count_nonzero(row) < cols and np.count_nonzero(row) > 0:
                    tensor[i, :] = row
                    break
        else:
            tensor[i, :] = 1
    np.random.shuffle(tensor)
    tensor = torch.tensor(tensor)
    return tensor

def add_noise(matrix, std, p):
    rows, cols = matrix.shape
    noisy_matrix = matrix.clone()
    for i in range(rows):
        if random.random() < p:
            noise = torch.randn(cols, device=device) * std
            noisy_matrix[i] += noise
    return noisy_matrix

dataset, _, view, data_size, class_num, dimss = load_data(args.dataset)
data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )
def AUMVC(iteration,miss_rate,noise_rate,Gaussian_noise):

    for batch_idx, (xs, y, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        break
    masked_xs = []
    noised_xs = []
    mix_xs = []

    num_rows = xs[0].shape[0]
    mask_tensor = mask(num_rows,view,miss_rate).to(device)


    for v in range(view):
        masked_x = mask_tensor[:,v].unsqueeze(1)*xs[v]
        masked_xs.append(masked_x)
    for v in range(view):
        noised_x = add_noise(xs[v],Gaussian_noise,noise_rate)
        noised_xs.append(noised_x)
    for v in range(view):
        mix_x = add_noise(masked_xs[v],Gaussian_noise,noise_rate)
        mix_xs.append(mix_x)
    xs_all = torch.cat(xs,dim=1)
    mask_all = torch.cat(masked_xs, dim=1)
    noise_all = torch.cat(noised_xs, dim=1)

    optimizer.zero_grad()
    xrs,_,xs_z,q = model(xs_all)
    mask_xrs,mask_h,mask_z,_ = model(mask_all)
    noise_xrs,noise_h,noise_z,_ = model(noise_all)

    loss_1 = criterion.alignment_loss(noise_z, mask_z)
    loss_2 = criterion.Uniformity_loss(noise_z,0.5)
    loss_3 = criterion.Uniformity_loss(mask_z,0.5)

    loss = loss_1 +0.5*(loss_2 + loss_3)

    loss.backward()
    optimizer.step()
    print('epoch {}'.format(iteration), 'Loss:{:.6f}'.format(loss))

accs = []
nmis = []
purs = []
aris = []

for i in range(T):
    print("ROUND:{}".format(i + 1))
    setup_seed(seed)
    model = Network(dimss, args.feature_dim, args.high_feature_dim,device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, device).to(device)

    miss_rate = args.miss_rate
    noise_rate = args.noise_rate
    Gaussian_noise = args.Gaussian_noise

    time0 = time.time()
    iteration = 1
    while iteration <= args.mse_iterations:

        AUMVC(iteration,miss_rate,noise_rate,Gaussian_noise)

        if iteration == args.mse_iterations:
                acc, nmi, ari, pur = valid(model, device, dataset, view, data_size, class_num, eval_z=True)
                accs.append(acc)
                nmis.append(nmi)
                purs.append(pur)
                aris.append(ari)
        iteration += 1


'''
print('%.4f'% np.mean(accs), '%.4f'% np.std(accs), accs)
print('%.4f'% np.mean(nmis), '%.4f'% np.std(nmis), nmis)
print('%.4f'% np.mean(aris), '%.4f'% np.std(aris), aris)
'''