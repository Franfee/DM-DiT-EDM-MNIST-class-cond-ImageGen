from torch.utils.data import DataLoader

from util.dataset import MNIST
from util.diffusion import Diffusion
import torch
from torch import nn
import os

from models.dit import DiT
from models.EDM_Precond import EDMPrecond
from models.EDM_Loss import EDMLoss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备
EPOCH = 100
BATCH_SIZE = 500


def train(loss_scaling=1, t=256):
    '''
        训练模型
    '''
    model.train()
    
    iter_count = 0
    for epoch in range(EPOCH):
        for _idx,(imgs, labels) in enumerate(dataloader):
            x = imgs * 2 - 1  # 图像的像素范围从[0,1]转换到[-1,1],和噪音高斯分布范围对应
            y = labels
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            loss = loss_fn(net=edm, img=x, condition=y)

            loss.sum().mul(loss_scaling / t).backward()
            optimzer.step()
            optimzer.zero_grad()
            
            if iter_count % 100 == 0:
                print('epoch:{} iter:{},loss:{}'.format(epoch, iter_count, loss.sum().mul(loss_scaling / t)))
            if iter_count % 2000 == 0:
                torch.save(edm, f'result/model/edm-{iter_count}.pth')

            iter_count += 1



if __name__ == '__main__':
    # 生成结果文件夹
    os.makedirs("result", exist_ok=True)
    os.makedirs(os.path.join("result", "model"), exist_ok=True)

    # 数据集
    dataset = MNIST()
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, persistent_workers=True)  # 数据加载器
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)  # 数据加载器

    # 模型
    model = DiT(img_size=28, patch_size=4, channel=1, emb_size=64, label_num=10, dit_num=3, head=4)
    edm = EDMPrecond(model=model).to(DEVICE)  # 模型

    # 优化器
    optimzer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 优化器
    loss_fn = EDMLoss()

    train()