import jittor as jt
import numpy as np
from model import StyledGenerator, VAE, PerceptualLoss, ZP2WPSpace, FeatExtract
import jittor.transform as transform
from dataset import SymbolDataset
from tqdm import tqdm
import argparse
import math
import random
from math import exp

jt.flags.use_cuda = True

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].update(par1[k] * decay + (1 - decay) * par2[k].detach())

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
    parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument('--ckpt', default=None, type=str, help='load from previous checkpoints')
    args = parser.parse_args()
    
    max_size  = 128
    max_step  = int(math.log2(max_size) - 2)

    batch_size = 16

    phase = 150_000
    max_iter = 10_000

    transform = transform.Compose([
        transform.ToPILImage(),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    lr = 1e-3

    step = 5
    netG = StyledGenerator(code_dim=512)
    vae = VAE(step + 1)
    feat_extract = FeatExtract()
    vae_optimizer = jt.optim.Adam(vae.parameters(), lr=lr, betas=(0.0, 0.99))
    '''
    zp2wpspace = ZP2WPSpace(step + 1)
    vae_optimizer.add_param_group(
        {'params':zp2wpspace.parameters(),
        'lr':lr,
        'betas':(0.0, 0.99)})
    '''
    ckpt = jt.load(args.ckpt)

    netG.load_state_dict(ckpt)
    p_loss = PerceptualLoss()
    print('Generator loaded')

    #saved = jt.load('FFHQ/checkpoint/vae_train_step-1500.model')
    #vae.load_state_dict(saved['vae'])
    ## Actual Training
    resolution = int(4 * 2 ** step)
    image_loader = SymbolDataset(args.path, transform, 1000).set_attrs(
        batch_size=batch_size, 
        shuffle=True
    )
    train_loader = iter(image_loader)
    print('image loader')

    #requires_grad(vae, True)
    requires_grad(netG, False)
    requires_grad(feat_extract, False)
    netG.eval()

    pbar = tqdm(range(max_iter))

    for batch in pbar:
        try:
            real_image = next(train_loader)
        except (OSError, StopIteration):
            train_loader = iter(image_loader)
            real_image = next(train_loader)
        real_image.requires_grad = True
        b_size = real_image.size(0)
        tot_loss = 0
        high, mid, low = feat_extract(real_image)
        mean, logvar = vae(high, mid, low)
        noises = []
        code = []
        kl_loss = jt.float32([0])
        print("mean ", mean[0].data.mean())
        print("var ", logvar[0].exp().data.mean())
        for i in range(step + 1):
            '''
            noise = jt.randn((b_size, 512))
            code.append(mean[i] + noise * (logvar[i] / 2).exp())
            kl_loss += -0.5 * (1 + logvar[i] - mean[i] * mean[i] - logvar[i].exp()).sum()
            '''
            code.append(mean[i])
            kl_loss += -0.5 * (1 - mean[i] * mean[i]).mean()
            #'''
        kl_loss *= 5e-4

        #code = zp2wpspace(code)
        opt = netG(code, step=step)
        l2_loss = jt.nn.MSELoss()(real_image, opt)
        per_loss = p_loss(real_image, opt)
        per_loss *= .05
        tot_loss += kl_loss
        tot_loss += l2_loss
        tot_loss += per_loss
        print("l2_loss ", l2_loss.data.sum(), "per_loss", per_loss.data.sum(), "kl_loss", kl_loss.data.sum())
        vae_optimizer.step(tot_loss)
        print("tot_loss ", tot_loss.data.sum())
        if batch % 500 == 0:
            ev = jt.contrib.concat([real_image, opt], 0)
            jt.save_image(ev, 'style_mixing/ev' + str(batch) + '.png', normalize=True, range=(-1, 1))
        if batch % 1000 == 0 and batch != 0:
            jt.save(
                {
                    'vae': vae.state_dict(),
                },
                f'FFHQ/checkpoint/vae_train_step-{batch}.model',
            )
