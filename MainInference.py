import torch

import matplotlib.pyplot as plt

from util.EDM_sampler import StackedRandomGenerator, edm_sampler


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备


edm = torch.load('result/model/edm-11000.pth', map_location=torch.device('cpu'))
edm = edm.to(DEVICE)

batch_size = 10
y = torch.arange(start=0, end=10, dtype=torch.long).to(DEVICE)

# seed
rnd = StackedRandomGenerator(DEVICE, y)
latents = rnd.randn([batch_size, 1, 28, 28], device=DEVICE)

steps = edm_sampler(net=edm, latents=latents, condition=y)

# 绘制数量
num_imgs = 20
# 绘制还原过程
plt.figure(figsize=(15, 15))
for b in range(batch_size):
    for i in range(0, num_imgs):
        idx = int(256 / num_imgs) * (i + 1)
        # 像素值还原到[0,1]
        final_img = (steps[idx][b].to('cpu') + 1) / 2
        # tensor转回PIL图
        final_img = final_img.permute(1, 2, 0)
        plt.subplot(batch_size, num_imgs, b * num_imgs + i + 1)
        plt.imshow(final_img)
plt.savefig("result/inf.png")
plt.show()