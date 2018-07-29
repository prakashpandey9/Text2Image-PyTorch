import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class Discriminator(nn.Module):
    def __init__(self, batch_size, img_size, text_embed_dim, text_reduced_dim):
        super(Discriminator, self).__init__()

        self.batch_size = batch_size
        self.img_size = img_size
        self.in_channels = img_size.size()[2]
        self.text_embed_dim = text_embed_dim
        self.text_reduced_dim = text_reduced_dim

        # Defining the discriminator network architecture
        self.d_net = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True))

        # output_dim = (batch_size, 4, 4, 512)
        # text.size() = (batch_size, text_embed_dim)

        # Defining a linear layer to reduce the dimensionality of caption embedding
        # from text_embed_dim to text_reduced_dim
        self.text_reduced_dim = nn.Linear(self.text_embed_dim, self.text_reduced_dim)

        self.cat_net = nn.Sequential(
            nn.Conv2d(512 + self.text_reduced_dim, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True))

        self.linear = nn.Linear(2 * 2 * 512, 1)

    def forward(self, image, text):
        """ Given the image and its caption embedding, predict whether the image
        is real or fake.

        Arguments
        ---------
        image : torch.FloatTensor
            image.size() = (batch_size, 64, 64, 3)

        text : torch.FloatTensor
            Output of the skipthought embedding model for the caption
            text.size() = (batch_size, text_embed_dim)

        --------
        Returns
        --------
        output : Probability for the image being real/fake
        logit : Final score of the discriminator

        """

        d_net_out = self.d_net(image)  # (batch_size, 4, 4, 512)
        text_reduced = self.text_reduced_dim(text)  # (batch_size, text_reduced_dim)
        text_reduced = text_reduced.squeeze(1)  # (batch_size, 1, text_reduced_dim)
        text_reduced = text_reduced.squeeze(2)  # (batch_size, 1, 1, text_reduced_dim)
        text_reduced = text_reduced.expand(1, 4, 4, self.text_reduced_dim)

        concat_out = torch.cat((d_net_out, text_reduced), 3)  # (1, 4, 4, 512+text_reduced_dim)

        logit = self.cat_net(concat_out)
        concat_out = torch.view(-1, concat_out.size()[1] * concat_out.size()[2] * concat_out.size()[3])
        concat_out = self.linear(concat_out)

        output = F.sigmoid(logit)

        return output, logit
