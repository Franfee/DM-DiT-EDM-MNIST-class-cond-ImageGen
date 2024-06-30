import torch
from tqdm import tqdm 


class Diffusion:
    def __init__(self, T, DEVICE="cuda"):
        self.T = T
        self.DEVICE = DEVICE
        # 前向diffusion计算参数
        self.betas=torch.linspace(0.0001,0.02,T) # (T,)
        self.alphas=1-self.betas  # (T,)
        self.alphas_cumprod=torch.cumprod(self.alphas,dim=-1) # alpha_t累乘 (T,)    [a1,a2,a3,....] ->  [a1,a1*a2,a1*a2*a3,.....]
        self.alphas_cumprod_prev=torch.cat((torch.tensor([1.0]),self.alphas_cumprod[:-1]),dim=-1) # alpha_t-1累乘 (T,),  [1,a1,a1*a2,a1*a2*a3,.....]
        self.variance=(1-self.alphas)*(1-self.alphas_cumprod_prev)/(1-self.alphas_cumprod)  # denoise用的方差   (T,)

    # 执行前向加噪
    def forward_add_noise(self,x,t): # batch_x: (batch,channel,height,width), batch_t: (batch_size,)
        noise=torch.randn_like(x)   # 为每张图片生成第t步的高斯噪音   (batch,channel,height,width)

        batch_alphas_cumprod=self.alphas_cumprod[t].view(x.size(0),1,1,1) 
        x=torch.sqrt(batch_alphas_cumprod)*x+torch.sqrt(1-batch_alphas_cumprod)*noise # 基于公式直接生成第t步加噪后图片
        return x,noise


    # 执行后向去噪
    def backward_denoise(self,model,x,y):
        steps=[x.clone(),]
        DEVICE = self.DEVICE
        
        model=model.to(DEVICE)
        x=x.to(DEVICE)
        y=y.to(DEVICE)
        
        self.betas=self.betas.to(DEVICE)
        self.alphas=self.alphas.to(DEVICE)
        self.alphas_cumprod=self.alphas_cumprod.to(DEVICE)
        self.alphas_cumprod_prev=self.alphas_cumprod_prev.to(DEVICE)
        self.variance=self.variance.to(DEVICE)

        model.eval()
        with torch.no_grad():
            for time in tqdm(range(self.T-1,-1,-1)):
                t=torch.full((x.size(0),),time).to(DEVICE) 

                # 预测x_t时刻的噪音
                noise=model(x,t,y)    
                
                # 生成t-1时刻的图像
                shape=(x.size(0),1,1,1) 
                mean=1/torch.sqrt(self.alphas[t].view(*shape))*  \
                    (
                        x-
                        (1-self.alphas[t].view(*shape))/torch.sqrt(1-self.alphas_cumprod[t].view(*shape))*noise
                    )
                
                if time!=0:
                    x=mean+ \
                        torch.randn_like(x)* \
                        torch.sqrt(self.variance[t].view(*shape))
                else:
                    x=mean
                
                x=torch.clamp(x, -1.0, 1.0).detach()
                steps.append(x)
        return steps