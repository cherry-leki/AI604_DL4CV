import os
import time
import datetime
import random
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from dataloader import create_dataloader, FolderDataset
from inception_score import Inception_Score
from util import conv, deconv, denorm, save_checkpoint, load_checkpoint


class CycleGAN_Generator(nn.Module):
    def __init__(self, input_nc: int=3, output_nc: int=3, ngf: int=64):
        """
        Parameters
        input_nc: the channels of input images
        output_nc: the channels of ouput images
        ngf: the numer of filters in the last convolutional layer
        """
        super(CycleGAN_Generator, self).__init__()

        # Architecture Hyper-parameters
        n_downsampling = 2
        n_blocks = 6
        model = []

        ### CycleGAN Generator
        # You have to implement CycleGAN Generation with 3 down-sampling layers, 6 residual blocks and 3 up-sampling layers.
        # For more details on the generator architecture, please check the homework description.
        # Note 1: Recommend to use 'conv' and 'deconv' function implemented in 'util.py'.
        # Note 2: You have to use 'nn.ReflectionPad2d' for padding.
        # Note 3: You have to use instance normalization for normalizing the feature maps.

        ### YOUR CODE HERE (~ 15 lines)
        # c7s1-k layer : 7x7 Convolution-InstanceNorm-ReLU layer with k filters and stride 1
        model.append(nn.ReflectionPad2d(3))
        model.append(conv(input_nc, ngf, k_size=7, stride=1, pad=0, bias=True, norm='in', activation='relu'))

        # ds layer : 3x3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model.append(conv(ngf * mult, ngf * mult * 2, k_size=3, stride=2, pad=1, bias=True, norm='in', activation='relu'))

        # ResNet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model.append(ResnetBlock(ngf * mult))

        # us layer : 3x3 fractional-strided-Convolution-InstanceNorm-ReLU layer with k filters and stride 1/2
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model.append(deconv(ngf * mult, int(ngf * (mult / 2)), k_size=3, stride=2, pad=1, output_padding=1, bias=True,
                                norm='in', activation='relu'))

        # c7s1-3 layer
        model.append(nn.ReflectionPad2d(3))
        model.append(conv(ngf, output_nc, k_size=7, stride=1, pad=0, norm=None, activation='tanh'))

        ### END YOUR CODE

        self.model = nn.Sequential(*model)

    def forward(self, input: torch.Tensor):
        return self.model(input)


class ResnetBlock(nn.Module):
    def __init__(self, dim: int):
        super(ResnetBlock, self).__init__()

        self.conv_block = None

        ### Residual block
        # You have to implement the residual block with 2 convolutional layers.
        # Note 1: Recommend to use 'conv' and 'deconv' function implemented in 'util.py'.
        # Note 2: You have to use 'nn.ReflectionPad2d' for padding.
        # Note 3: You have to use instance normalization for normalizing the feature maps.

        ### YOUR CODE HERE (~ 4 lines)
        conv_block = [nn.ReflectionPad2d(1)]
        conv_block += [conv(dim, dim, k_size=3, stride=1, pad=0, bias=True, norm='in', activation='relu')]
        conv_block += [(nn.ReflectionPad2d(1))]
        conv_block += [conv(dim, dim, k_size=3, stride=1, pad=0, bias=True, norm='in', activation=None)]

        self.conv_block = nn.Sequential(*conv_block)
        ### END YOUR CODE

    def forward(self, x: torch.Tensor):
        out = None

        ### Skip connection
        # Implement the skip connection

        # Add skip connections
        ### YOUR CODE HERE (~ 1 line)
        out = x + self.conv_block(x)

        ### END YOUR CODE
        return out


class CycleGAN_Discriminator(nn.Module):
    def __init__(self, input_nc: int=3, ndf: int=64):
        """
        Parameters
        input_nc: the channels of input images
        ndf: the numer of filters in the first convolutional layer
        """
        super(CycleGAN_Discriminator, self).__init__()

        # Architecture Hyper-parameters
        n_layers = 3
        model = []

        ### CycleGAN Discriminator
        # You have to implement the CycleGAN Discriminator.
        # For more details on the discriminator architecture, please check the homework description.
        # Note 1: Recommend to use 'conv' and 'deconv' function implemented in 'util.py'.
        # Note 2: You have to use instance normalization for normalizing the feature maps.

        ### YOUR CODE HERE (~ 12 lines)
        model.append(conv(input_nc, ndf, k_size=4, stride=2, pad=1, norm=None, activation='lrelu'))

        ndf_mult_size = ndf
        for i in range(n_layers-1):
            model.append(conv(ndf_mult_size, ndf_mult_size * 2, k_size=4, stride=2, pad=1, bias=True, norm='in', activation='lrelu'))
            ndf_mult_size = ndf_mult_size * 2

        model.append(conv(ndf_mult_size, ndf_mult_size * 2, k_size=4, stride=1, pad=1, bias=True, norm='in', activation='lrelu'))
        ndf_mult_size = ndf_mult_size * 2
        model.append(conv(ndf_mult_size, 1, k_size=4, stride=1, pad=1, norm=None, activation=None))

        ### END YOUR CODE

        self.model = nn.Sequential(*model)

    def forward(self, input: torch.Tensor):
        return self.model(input)


class CycleGAN_Solver():
    def __init__(self, lr: float=0.0002, batch_size: int=1, num_workers: int=1):
        """
        Parameters
        lr: learning rate
        batch_size: batch size
        num_workers: the number of workers for train dataloader
        """

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Declare Generator and Discriminator
        self.netG_A2B = CycleGAN_Generator(input_nc=3, output_nc=3, ngf=64)
        self.netG_B2A = CycleGAN_Generator(input_nc=3, output_nc=3, ngf=64)
        self.netD_A = CycleGAN_Discriminator(input_nc=3, ndf=64)
        self.netD_B = CycleGAN_Discriminator(input_nc=3, ndf=64)

        # Declare the Criterion for GAN loss
        # Doc for MSE Loss: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
        # Doc for L1 Loss: https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html

        self.criterion_GAN: nn.Module = None
        self.criterion_cycle: nn.Module = None
        self.criterion_identity: nn.Module = None

        ### YOUR CODE HERE (~ 3 lines)
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        ### END YOUR CODE

        self.optimizerG: optim.Optimizer = None
        self.optimizerD_A: optim.Optimizer = None
        self.optimizerD_B: optim.Optimizer = None

        # Declare the Optimizer for training
        # Doc for Adam optimizer: https://pytorch.org/docs/stable/optim.html#torch.optim.Adam

        ### YOUR CODE HERE (~ 3 lines)
        self.optimizerG = optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()), lr=lr)
        self.optimizerD_A = optim.Adam(self.netD_A.parameters(), lr=lr)
        self.optimizerD_B = optim.Adam(self.netD_B.parameters(), lr=lr)

        ### END YOUR CODE

        # Declare the DataLoader
        # You have to implement 'Summer2WinterDataset' in 'dataloader.py'
        # Note1: Use 'create_dataloader' function implemented in 'dataloader.py'

        ### YOUR CODE HERE (~ 1 line)
        self.trainloader, self.testloader = create_dataloader(dataset='summer2winter', batch_size=batch_size,
                                                              num_workers=num_workers)

        ### END YOUR CODE

        # Make directory
        os.makedirs('./results/cyclegan/images/', exist_ok=True)
        os.makedirs('./results/cyclegan/checkpoints/', exist_ok=True)

    def train(self, epochs: int=100):

        self.netG_A2B.to(self.device)
        self.netG_B2A.to(self.device)
        self.netD_A.to(self.device)
        self.netD_B.to(self.device)

        start_time = time.time()

        print("=====Train Start======")

        for epoch in range(epochs):
            for iter, (real_A, real_B) in enumerate(self.trainloader):
                self.netG_A2B.train()
                self.netG_B2A.train()
                self.netD_A.train()
                self.netD_B.train()

                real_A = real_A.to(self.device)
                real_B = real_B.to(self.device)

                ###################################################################################
                # (1) Update Generator
                # Train the generator (self.netG_A2B & self.netG_B2A).
                # You have to implement 3 types of loss functions ('identity loss', 'gan loss', 'cycle consistency loss')
                ###################################################################################

                identity_loss: torch.Tensor() = None
                gan_loss: torch.Tensor() = None
                cycle_loss: torch.Tensor() = None
                lossG: torch.Tensor() = None

                ### YOUR CODE HERE (~ 15 lines)
                for param in [self.netD_A, self.netD_B]:
                    param.requires_grad_(False)

                # identity_loss
                idt_B = self.netG_A2B(real_B)   # G_A2B(B) should be equal B if real B is fed
                idt_A = self.netG_B2A(real_A)   # G_B2A(A) should be equal A if real A is fed

                # loss_idt_B + loss_idt_A
                identity_loss = self.criterion_identity(idt_B, real_B) + self.criterion_identity(idt_A, real_A)
                identity_loss = 5 * identity_loss

                # gan_loss
                fake_B = self.netG_A2B(real_A)  # G_A2B(A_real): Y = G(X)
                fake_A = self.netG_B2A(real_B)  # G_B2A(B_real): X = F(Y)

                fake_B_D = self.netD_B(fake_B)
                fake_A_D = self.netD_A(fake_A)

                label_real = torch.ones(fake_B_D.size()).to(self.device)
                label_fake = torch.zeros(fake_B_D.size()).to(self.device)
                # loss_gan_B + loss_gan_A
                gan_loss = self.criterion_GAN(fake_B_D, label_real) + self.criterion_GAN(fake_A_D, label_real)

                # cycle_loss
                recon_A = self.netG_B2A(fake_B)  # G_B2A(B_fake): X' = F(G(X))
                recon_B = self.netG_A2B(fake_A)  # G_A2B(A_fake): Y' = G(F(Y))

                # loss_cycle_A + loss_cycle_B
                cycle_loss = self.criterion_cycle(recon_A, real_A) + self.criterion_cycle(recon_B, real_B)
                cycle_loss = 10 * cycle_loss

                lossG = identity_loss + gan_loss + cycle_loss
                ### END YOUR CODE

                # Test code
                if epoch == 0 and iter == 0:
                    test_lossG_fuction(identity_loss, gan_loss, cycle_loss)

                self.optimizerG.zero_grad()
                lossG.backward()
                self.optimizerG.step()

                ###################################################################################
                # (2) Update Discriminator
                # Train the discrminator (self.netD_A & self.netD_B).
                ###################################################################################
                # Discriminator A

                lossD_A: torch.Tensor() = None

                ### YOUR CODE HERE (~ 4 lines)
                for param in [self.netD_A, self.netD_B]:
                    param.requires_grad_(True)
                pred_A_real = self.netD_A(real_A)
                pred_A_fake = self.netD_A(fake_A.detach())
                lossD_A = (self.criterion_GAN(pred_A_fake, label_fake) + self.criterion_GAN(pred_A_real, label_real))

                ### END YOUR CODE

                self.optimizerD_A.zero_grad()
                lossD_A.backward()
                self.optimizerD_A.step()

                # Discriminator B

                lossD_B: torch.Tensor() = None

                ### YOUR CODE HERE (~ 4 lines)
                pred_B_real = self.netD_B(real_B)
                pred_B_fake = self.netD_B(fake_B.detach())
                lossD_B = (self.criterion_GAN(pred_B_fake, label_fake) + self.criterion_GAN(pred_B_real, label_real))

                ### END YOUR CODE

                # Test code
                if epoch == 0 and iter == 0:
                    test_lossD_fuction(lossD_A, lossD_B)

                self.optimizerD_B.zero_grad()
                lossD_B.backward()
                self.optimizerD_B.step()

                if (iter + 1) % 100 == 0:
                    end_time = time.time() - start_time
                    end_time = str(datetime.timedelta(seconds=end_time))[:-7]
                    print('Time [%s], Epoch [%d/%d], Step[%d/%d], lossD_A: %.4f, lossD_B: %.4f, lossG: %.4f'
                          % (end_time, epoch+1, epochs, iter+1, len(self.trainloader), lossD_A.item(), lossD_A.item(), lossG.item()))

            # Save Images
            fake_A = fake_A.reshape(fake_A.size(0), 3, 256, 256)
            fake_B = fake_B.reshape(fake_B.size(0), 3, 256, 256)

            save_image(denorm(fake_B), os.path.join('./results/cyclegan/images', 'fakeA2B-{:03d}.png'.format(epoch + 1)))
            save_image(denorm(fake_A), os.path.join('./results/cyclegan/images', 'fakeB2A-{:03d}.png'.format(epoch + 1)))

            if (epoch + 1) % 10 == 0:
                save_checkpoint(self.netG_A2B, './results/cyclegan/checkpoints/netG_A2B_{:02d}.pth'.format(epoch+1), self.device)
                save_checkpoint(self.netG_B2A, './results/cyclegan/checkpoints/netG_B2A_{:02d}.pth'.format(epoch+1), self.device)

                save_checkpoint(self.netD_A, './results/cyclegan/checkpoints/netD_A_{:02d}.pth'.format(epoch+1), self.device)
                save_checkpoint(self.netD_B, './results/cyclegan/checkpoints/netD_B_{:02d}.pth'.format(epoch+1), self.device)


        # Save Checkpoints
        save_checkpoint(self.netG_A2B, './results/cyclegan/checkpoints/netG_A2B_final.pth', self.device)
        save_checkpoint(self.netG_B2A, './results/cyclegan/checkpoints/netG_B2A_final.pth', self.device)

        save_checkpoint(self.netD_A, './results/cyclegan/checkpoints/netD_A_final.pth', self.device)
        save_checkpoint(self.netD_B, './results/cyclegan/checkpoints/netD_B_final.pth', self.device)


    def test(self):
        load_checkpoint(self.netG_A2B, './results/cyclegan/checkpoints/netG_A2B_final.pth', self.device)
        load_checkpoint(self.netG_B2A, './results/cyclegan/checkpoints/netG_B2A_final.pth', self.device)

        os.makedirs('./results/cyclegan/evaluation/', exist_ok=True)

        self.netG_A2B.eval()
        self.netG_B2A.eval()

        print("=====Test Start======")

        with torch.no_grad():
            for iter, (real_A, real_B) in enumerate(self.testloader):
                fake_A = self.netG_B2A(real_B.to(self.device))
                save_image(denorm(fake_A), os.path.join('./results/cyclegan/evaluation', 'fake_image-{:05d}.png'.format(iter + 1)))

        # Compute the Inception score
        dataset = FolderDataset(folder=os.path.join('./results/cyclegan/evaluation'))
        Inception = Inception_Score(dataset)
        score = Inception.compute_score(splits=1)

        print('Inception Score : ', score)


#############################################
# Testing functions below.                  #
#############################################

def test_initializer_and_forward():

    print("=====Model Initializer Test Case======")

    netG = CycleGAN_Generator(input_nc=3, output_nc=3, ngf=64)

    try:
        netG.load_state_dict(torch.load("sanity_check/sanity_check_cyclegan_netG.pth", map_location='cpu'))
    except Exception as e:
        print("Your CycleGAN generator initializer is wrong. Check the handout and comments in details and implement the model precisely.")
        raise e
    print("The first test passed!")

    netD = CycleGAN_Discriminator(input_nc=3, ndf=64)

    try:
        netD.load_state_dict(torch.load("sanity_check/sanity_check_cyclegan_netD.pth", map_location='cpu'))
    except Exception as e:
        print("Your CycleGAN discriminator initializer is wrong. Check the handout and comments in details and implement the model precisely.")
        raise e
    print("The second test passed!")

    print("All 2 tests passed!")


def test_lossG_fuction(identity_loss, gan_loss, cycle_loss):

    print("=====Generator Loss Test Case======")

    # the first test
    assert identity_loss.detach().allclose(torch.tensor(5.7538), atol=1e-2), \
        f"Identity Loss of the model does not match expected result."
    print("The first test passed!")

    # the second test
    assert gan_loss.detach().allclose(torch.tensor(2.2067), atol=1e-2), \
        f"Adversarial Loss of the model does not match expected result."
    print("The second test passed!")

    # the third test
    assert cycle_loss.detach().allclose(torch.tensor(11.5476), atol=1e-2), \
        f"Cycle Consistency Loss of the model does not match expected result."
    print("The third test passed!")

    print("All test passed!")

def test_lossD_fuction(lossD_A, lossD_B):

    print("=====Discriminator Loss Test Case======")

    assert lossD_A.detach().allclose(torch.tensor(1.1283), atol=1e-2), \
        f"Discriminator A Loss of the model does not match expected result."
    print("The first test passed!")

    assert lossD_B.detach().allclose(torch.tensor(1.2319), atol=1e-2), \
        f"Discriminator B Loss of the model does not match expected result."
    print("The second test passed!")

    print("All test passed!")


if __name__ == "__main__":
    torch.set_printoptions(precision=4)
    random.seed(1234)
    torch.manual_seed(1234)

    test_initializer_and_forward()

    # Hyper-parameters
    # For good performance, you have to train CycleGAN about 200 epochs.
    # However, it takes 1 to 2 days to train the model, so it is okay to train only 30 epochs.
    epochs = 30 # 200
    lr = 0.0002
    batch_size = 1
    num_workers = 1
    train = True  # train : True / test : False (Compute the Inception Score)

    # Train or Test
    solver = CycleGAN_Solver(lr=lr, batch_size=batch_size, num_workers=num_workers)
    if train:
        solver.train(epochs=epochs)
    else:
        solver.test()