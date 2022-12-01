
import os
import torch
import joblib
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from tools.classify_result import classify_result

def parse_args():
    parser = argparse.ArgumentParser()

    # basic parameters: Train
    parser.add_argument('--output_path', type=str, default=r'.\result\AAU_Net', help='the output path')
    parser.add_argument('--data_path', type=str, default=r'.\data\Simulated_dataset.pkl', help='the output path')
    parser.add_argument("--n_epochs", type=int, default=15, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr_g", type=float, default=0.0005, help="learning rate of generator")
    parser.add_argument("--lr_d", type=float, default=0.0002, help="learning rate of discriminator")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--use_cuda", type=bool, default=True, help="whether to sue cuda")

    # hyper-parameters: Network
    parser.add_argument('--IC', type=list, default=[1, 1], help='input channels of the networks')
    parser.add_argument('--CL', type=list, default=[32, 64, 128, 512], help='channels list of the networks')
    parser.add_argument('--KL', type=list, default=[8, 8, 4, 4], help='kernals list of the networks')
    parser.add_argument('--PL', type=list, default=[2, 2, 1, 1], help='paddings list of the networks')
    parser.add_argument('--SL', type=list, default=[4, 4, 2, 2], help='strides list of the networks')
    parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent code")
    parser.add_argument('--unfoldings', type=int, default=4, help='unrollings of the networks')
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--wa", type=float, default=0.001, help="the weight for the adversarial loss")
    parser.add_argument("--wr", type=float, default=0.999, help="the weight for the reconstruction loss")
    parser.add_argument("--ratio", type=float, default=0.333, help="the rough ratio of the anomaly samples")
    opt = parser.parse_args(args=[])

    return opt

class SoftThreshold_1d(nn.Module):
    def __init__(self, channel_num, init_threshold=1e-3):
        super(SoftThreshold_1d, self).__init__()
        self.threshold = nn.Parameter(init_threshold * torch.ones(1,channel_num,1))

    def forward(self, x):
        mask1 = (x > self.threshold).float()
        mask2 = (x < -self.threshold).float()
        out = mask1.float() * (x - self.threshold)
        out += mask2.float() * (x + self.threshold)
        return out

class UnrolledAutoEncoder(nn.Module):
    """
    NetG NETWORK
    """
    def __init__(self, opt):
        super(UnrolledAutoEncoder, self).__init__()
        self.opt = opt
        self.opt.Tensor = torch.cuda.FloatTensor if opt.use_cuda else torch.FloatTensor
        self.T = opt.unfoldings
        # Convolutional Filters
        self.W1 = nn.Parameter(torch.randn(opt.CL[0], opt.IC[0], opt.KL[0]), requires_grad=True)
        self.strd1 = opt.SL[0]; self.pad1 = opt.PL[0]

        self.W2 = nn.Parameter(torch.randn(opt.CL[1], opt.CL[0], opt.KL[1]), requires_grad=True)
        self.strd2 = opt.SL[1]; self.pad2 = opt.PL[1]

        self.W3 = nn.Parameter(torch.randn(opt.CL[2], opt.CL[1], opt.KL[2]), requires_grad=True)
        self.strd3 = opt.SL[2]; self.pad3 = opt.PL[2]

        self.W4 = nn.Parameter(torch.randn(opt.CL[3], opt.CL[2], opt.KL[3]), requires_grad=True)
        self.strd4 = opt.SL[3]; self.pad4 = opt.PL[3]

        self.c1 = nn.Parameter(torch.ones(1, 1, 1), requires_grad=True)
        self.c2 = nn.Parameter(torch.ones(1, 1, 1), requires_grad=True)
        self.c3 = nn.Parameter(torch.ones(1, 1, 1), requires_grad=True)
        self.c4 = nn.Parameter(torch.ones(1, 1, 1), requires_grad=True)

        # linear
        self.mu     = nn.Linear(self.opt.CL[3]*16, self.opt.latent_dim)
        self.logvar = nn.Linear(self.opt.CL[3]*16, self.opt.latent_dim)
        self.linear = nn.Linear(self.opt.latent_dim, self.opt.CL[3]*16)


        # Biases / Thresholds
        self.soft1 = SoftThreshold_1d(opt.CL[0])
        self.soft2 = SoftThreshold_1d(opt.CL[1])
        self.soft3 = SoftThreshold_1d(opt.CL[2])
        self.soft4 = SoftThreshold_1d(opt.CL[3])

        # Initialization
        self.W1.data = .1 / np.sqrt(opt.IC[0] * opt.KL[0]) * self.W1.data
        self.W2.data = .1 / np.sqrt(opt.CL[0] * opt.KL[1]) * self.W2.data
        self.W3.data = .1 / np.sqrt(opt.CL[1] * opt.KL[2]) * self.W3.data
        self.W4.data = .1 / np.sqrt(opt.CL[2] * opt.KL[3]) * self.W4.data

    def forward(self, x, test=False):
        # Encoding
        gamma1 = self.soft1(self.c1 * F.conv1d(x,      self.W1, stride=self.strd1, padding=self.pad1))
        gamma2 = self.soft2(self.c2 * F.conv1d(gamma1, self.W2, stride=self.strd2, padding=self.pad2))
        gamma3 = self.soft3(self.c3 * F.conv1d(gamma2, self.W3, stride=self.strd3, padding=self.pad3))
        gamma4 = self.soft4(self.c4 * F.conv1d(gamma3, self.W4, stride=self.strd4, padding=self.pad4))

        for _ in range(self.T):
            # forward computation: gamma(i+1) = soft(gamma^(i+1)-c*DT*(D*gamma^(i+1)-gamma(i)))
            gamma1 = self.soft1((gamma1 - self.c1 * F.conv1d(F.conv_transpose1d(gamma1, self.W1, stride=self.strd1, padding=self.pad1) - x, self.W1, stride=self.strd1, padding=self.pad1)))
            gamma2 = self.soft2((gamma2 - self.c2 * F.conv1d(F.conv_transpose1d(gamma2, self.W2, stride=self.strd2, padding=self.pad2) - gamma1, self.W2, stride=self.strd2, padding=self.pad2)))
            gamma3 = self.soft3((gamma3 - self.c3 * F.conv1d(F.conv_transpose1d(gamma3, self.W3, stride=self.strd3, padding=self.pad3) - gamma2, self.W3, stride=self.strd3, padding=self.pad3)))
            gamma4 = self.soft4((gamma4 - self.c4 * F.conv1d(F.conv_transpose1d(gamma4, self.W4, stride=self.strd4, padding=self.pad4) - gamma3, self.W4, stride=self.strd4, padding=self.pad4)))

        # Calculate the paramater
        mu =     self.mu(gamma4.view(gamma4.shape[0],-1))
        logvar = self.logvar(gamma4.view(gamma4.shape[0],-1))
        z = reparameterization(mu, logvar, self.opt)

        # Decoding
        if test:
            gamma4_hat = self.linear(mu).view_as(gamma4)
            z = mu
        else:
            gamma4_hat = self.linear(z).view_as(gamma4)
        gamma3_hat = F.conv_transpose1d(gamma4_hat, self.W4, stride=self.strd4, padding=self.pad4)
        gamma2_hat = F.conv_transpose1d(gamma3_hat, self.W3, stride=self.strd3, padding=self.pad3)
        gamma1_hat = F.conv_transpose1d(gamma2_hat, self.W2, stride=self.strd2, padding=self.pad2)
        x_hat      = F.conv_transpose1d(gamma1_hat, self.W1, stride=self.strd1, padding=self.pad1)
        return x_hat, z

def reparameterization(mu, logvar, opt):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(opt.Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
    z = sampled_z * std + mu
    return z

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.opt = opt
        self.model = nn.Sequential(
            nn.Linear(self.opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity

def train_AAUNet(opt):

    # Use binary cross-entropy loss
    adversarial_loss = torch.nn.BCELoss()
    pixelwise_loss = torch.nn.L1Loss()

    # Initialize generator and discriminator
    generator     = UnrolledAutoEncoder(opt)
    discriminator = Discriminator(opt)

    if opt.use_cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        pixelwise_loss.cuda()

    # Configure data loader
    outf = opt.output_path
    os.makedirs(outf, exist_ok=True)
    os.makedirs(os.path.join(outf, 'model'), exist_ok=True)
    data_dict = joblib.load(opt.data_path)
    train = TensorDataset(torch.Tensor(data_dict['train_1d']), torch.Tensor(data_dict['train_label']))
    test =  TensorDataset(torch.Tensor(data_dict['test_1d']), torch.Tensor(data_dict['test_label']))
    dataloader = torch.utils.data.DataLoader(dataset=train, batch_size=opt.batch_size, shuffle=True, drop_last=False)
    testloader = torch.utils.data.DataLoader(dataset=test,  batch_size=opt.batch_size, shuffle=True, drop_last=False)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))
    
    #  Training
    ACC, TPR, FPR = np.zeros((opt.n_epochs,)), np.zeros((opt.n_epochs,)), np.zeros((opt.n_epochs,))
    opt.Tensor = torch.cuda.FloatTensor if opt.use_cuda else torch.FloatTensor
    with open(os.path.join(outf,"train_log.txt"), "w") as f:
        for epoch in range(opt.n_epochs):

            generator.train()
            discriminator.train()

            for i, (x, _) in enumerate(dataloader):

                # Adversarial ground truths
                valid = Variable(opt.Tensor(x.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(opt.Tensor(x.shape[0], 1).fill_(0.0), requires_grad=False)

                # Configure input
                x_real = Variable(x.type(opt.Tensor))

                #  Train Generator
                optimizer_G.zero_grad()
                x_hat, x_z = generator(x_real)

                # Loss measures generator's ability to fool the discriminator
                g_loss = opt.wa * adversarial_loss(discriminator(x_z), valid) + opt.wr * pixelwise_loss(x_hat, x_real)

                g_loss.backward()
                optimizer_G.step()

                #  Train Discriminator
                if i % opt.n_critic==0:
                    z = Variable(opt.Tensor(np.random.normal(0, 1, (x_hat.shape[0], opt.latent_dim))))
                    optimizer_D.zero_grad()
                    # Measure discriminator's ability to classify real from generated samples
                    real_loss = adversarial_loss(discriminator(z), valid)
                    fake_loss = adversarial_loss(discriminator(x_z.detach()), fake)
                    d_loss = 0.5 * (real_loss + fake_loss)

                    d_loss.backward()
                    optimizer_D.step()
            
            # testing
            generator.eval()
            discriminator.eval()
            TP, TN, FP, FN = 0, 0, 0, 0
            for i, (x, y) in enumerate(testloader):
                x_real = Variable(x.type(opt.Tensor))
                x_hat, x_z = generator(x_real)
                score = discriminator(x_z)
                result = score.cpu().detach().numpy()
                right_i, TP_i, TN_i, FP_i, FN_i = classify_result(result, y.cpu().detach().numpy(), print_result=False)
                TP+=TP_i.shape[0]; TN+=TN_i.shape[0]; FP+=FP_i.shape[0]; FN+=FN_i.shape[0]
            
            ACC[epoch] = 100*float(TP+TN)/float(len(testloader.dataset))
            TPR[epoch], FPR[epoch] = 100 * float(TP) / float(TP + FN + 0.00001), 100 * float(FP) / float(TN + FP + 0.00001)
            print("Epoch: %d/%d | G loss: %f | D loss: %f | ACC: %f | TPR: %f | FPR: %f" % (epoch+1, opt.n_epochs, g_loss.item(), d_loss.item(), ACC[epoch], TPR[epoch], FPR[epoch]))
            # save models
            torch.save(generator.state_dict(), '%s/model_epo_%03d_GLoss_%.4f_DLoss_%.4f_Generator.pth' % (os.path.join(outf,'model'), epoch+1, g_loss.item(), d_loss.item()))
            torch.save(discriminator.state_dict(), '%s/model_epo_%03d_GLoss_%.4f_DLoss_%.4f_Discrim.pth' % (os.path.join(outf,'model'), epoch+1, g_loss.item(), d_loss.item()))
            f.write("EPOCH = %03d, G_Loss: %.8f, D_Loss: %.8f, ACC: %f, TPR: %f, FPR: %f" %(epoch+1, g_loss.item(), d_loss.item(), ACC[epoch], TPR[epoch], FPR[epoch]))
            f.write('\n')
            f.flush()
    f.close()
    return ACC