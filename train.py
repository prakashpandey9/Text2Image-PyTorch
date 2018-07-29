import os
import args
import time
import datetime
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image

import numpy as np
from nets import Discriminator, Generator


class GAN_CLS(object):
    def __init__(self, args, data_loader, SUPERVISED=True):
        """
        Arguments :
        ----------
        args : Arguments defined in Argument Parser
        data_loader = An instance of class DataLoader for loading our dataset in batches
        SUPERVISED :

        """

        self.data_loader = data_loader
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size

        self.log_step = config.log_step
        self.sample_step = config.sample_step

        self.log_dir = args.log_dir
        self.checkpoint_dir = args.checkpoint_dir
        self.sample_dir = config.sample_dir
        self.final_model = args.final_model

        self.dataset = args.dataset
        self.model_name = args.model_name

        self.img_size = args.img_size
        self.z_dim = args.z_dim
        self.text_embed_dim = args.text_embed_dim
        self.text_reduced_dim = args.text_reduced_dim
        self.learning_rate = args.learning_rate
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.l1_coeff = args.l1_coeff
        self.resume_epoch = args.resume_epoch
        self.SUPERVISED = SUPERVISED

        # Logger setting
        self.logger = logging.getLogger('__name__')
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        self.file_handler = logging.FileHandler(self.log_dir)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

        self.build_model()

    def build_model(self):
        """ A function of defining following instances :

        -----  Generator
        -----  Discriminator
        -----  Optimizer for Generator
        -----  Optimizer for Discriminator
        -----  Defining Loss functions

        """

        # ---------------------------------------------------------------------
        #						1. Network Initialization
        # ---------------------------------------------------------------------
        self.gen = Generator(batch_size=self.batch_size,
                             img_size=self, img_size,
                             z_dim=self.z_dim,
                             text_embed_dim=self.text_embed_dim,
                             text_reduced_dim=self.text_reduced_dim)

        self.disc = Discriminator(batch_size=self.batch_size,
                                  img_size=self, img_size,
                                  text_embed_dim=self.text_embed_dim,
                                  text_reduced_dim=self.text_reduced_dim)

        self.gen_optim = optim.Adam(self.gen.parameters(),
                                    lr=self.learning_rate,
                                    betas=(self.beta1, self.beta2))

        self.disc_optim = optim.Adam(self.disc.parameters(),
                                     lr=self.learning_rate,
                                     betas=(self.beta1, self.beta2))

        self.cls_gan_optim = optim.Adam(itertools.chain(self.gen.parameters(),
                                                        self.disc.parameters()),
                                        lr=self.learning_rate,
                                        betas=(self.beta1, self.beta2))

        print ('-------------  Generator Model Info  ---------------')
        self.print_network(self.gen, 'G')
        print ('------------------------------------------------')

        print ('-------------  Discriminator Model Info  ---------------')
        self.print_network(self.disc, 'D')
        print ('------------------------------------------------')

        self.gen.cuda()
        self.disc.cuda()
        self.criterion = nn.BCELoss().cuda()
        # self.CE_loss = nn.CrossEntropyLoss().cuda()
        # self.MSE_loss = nn.MSELoss().cuda()
        self.gen.train()
        self.disc.train()

    def print_network(self, model, name):
        """ A function for printing total number of model parameters """
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()

        print(model)
        print(name)
        print("Total number of parameters: {}".format(num_params))

    def load_checkpoints(self, resume_epoch):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_epoch))
        G_path = os.path.join(self.checkpoint_dir, '{}-G.ckpt'.format(resume_epoch))
        D_path = os.path.join(self.checkpoint_dir, '{}-D.ckpt'.format(resume_epoch))
        self.gen.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.disc.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def train_model(self):

        data_loader = self.data_loader

        start_epoch = 0
        if self.resume_epoch:
            start_epoch = self.resume_epoch
            self.load_checkpoints(self.resume_epoch)

        print ('---------------  Model Training Started  ---------------')
        start_time = time.time()

        for epoch in range(start_epoch, self.num_epochs):
            for idx, batch in enumerate(data_loader):
                true_imgs = batch['true_imgs']
                true_embed = batch['true_embed']
                false_imgs = batch['false_imgs']

                real_labels = torch.ones(true_imgs.size(0))
                fake_labels = torch.zeros(true_imgs.size(0))

                smooth_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.1))

                true_imgs = Variable(true_imgs.float()).cuda()
                true_embed = Variable(true_embed.float()).cuda()
                false_imgs = Variable(false_imgs.float()).cuda()

                real_labels = Variable(real_labels).cuda()
                smooth_real_labels = Variable(smooth_real_labels).cuda()
                fake_labels = Variable(fake_labels).cuda()

                # ---------------------------------------------------------------
                # 					  2. Training the generator
                # ---------------------------------------------------------------
                self.gen.zero_grad()
                z = Variable(torch.randn(true_imgs.size(0), self.z_dim)).cuda()
                fake_imgs = self.gen(true_embed, z)
                fake_out, fake_logit = self.disc(fake_imgs, true_embed)
                true_out, true_logit = self.disc(true_imgs, true_embed)

                gen_loss = self.criterion(fake_out, real_labels) +
                    self.l1_coeff * nn.L1Loss(fake_imgs, true_imgs)

                gen_loss.backward()
                self.gen_optim.step()

                # ---------------------------------------------------------------
                # 					3. Training the discriminator
                # ---------------------------------------------------------------
                self.disc.zero_grad()
                false_out, false_logit = self.disc(false_imgs, true_embed)
                disc_loss = self.criterion(true_out, smooth_real_labels) +
                    self.criterion(fake_out, fake_labels) + self.criterion(false_out, fake_labels)

                disc_loss.backward()
                self.disc_optim.step()

                # self.cls_gan_optim.step()

                # Logging
                loss = {}
                loss['G_loss'] = gen_loss.item()
                loss['D_loss'] = disc_loss.item()

                # ---------------------------------------------------------------
                # 					4. Logging INFO into log_dir
                # ---------------------------------------------------------------
                if (idx + 1) % self.log_step == 0:
                    end_time = time.time() - start_time
                    end_time = datetime.timedelta(seconds=end_time)
                    log = "Elapsed [{}], Epoch [{}/{}], Idx [{}]".format(end_time, epoch + 1,
                                                                         self.num_epochs, idx)

                for net, loss_value in loss.items():
                    log += ", {}: {:.4f}".format(net, loss_value)
                    self.logger.info(log)
                    print (log)

                # ---------------------------------------------------------------
                # 					5. Saving generated images
                # ---------------------------------------------------------------
                if (idx + 1) % self.sample_step == 0:
                    concat_imgs = torch.cat((true_imgs, fake_imgs), 2)  # ??????????
                    save_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(idx + 1))
                    cocat_imgs = (cocat_imgs + 1) / 2
                    # out.clamp_(0, 1)
                    save_image(concat_imgs.data.cpu(), self.sample_dir, nrow=1, padding=0)
                    print ('Saved real and fake images into {}...'.format(self.sample_dir))

                # ---------------------------------------------------------------
                # 				6. Saving the checkpoints & final model
                # ---------------------------------------------------------------
                if (idx + 1) % self.model_save_step == 0:
                    G_path = os.path.join(self.checkpoint_dir, '{}-G.ckpt'.format(idx + 1))
                    D_path = os.path.join(self.checkpoint_dir, '{}-D.ckpt'.format(idx + 1))
                    torch.save(self.gen.state_dict(), G_path)
                    torch.save(self.disc.state_dict(), D_path)
                    print('Saved model checkpoints into {}...'.format(self.checkpoint_dir))

        print ('---------------  Model Training Completed  ---------------')
        # Saving final model into final_model directory
        G_path = os.path.join(self.final_model, '{}-G.pth'.format('final'))
        D_path = os.path.join(self.final_model, '{}-D.pth'.format('final'))
        torch.save(self.gen.state_dict(), G_path)
        torch.save(self.disc.state_dict(), D_path)
        print('Saved final model into {}...'.format(self.final_model))
