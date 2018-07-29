import os
import time
import argsparse
import skipthoughts

import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image

from net import Generator


def test():

    parser = argsparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--img_size', type=int, default=64,
                        help='Size of the image')
    parser.add_argument('--z_dim', type=int, default=100,
                        help='Size of the latent variable')
    parser.add_argument('--final_model', type=str, default='final_model',
                        help='Save INFO into logger after every x iterations')
    parser.add_argument('--save_img', type=str, default='test',
                        help='Save predicted images')
    parser.add_argument('--text_embed_dim', type=int, default=4800,
                        help='Size of the embeddding for the captions')
    parser.add_argument('--text_reduced_dim', type=int, default=1024,
                        help='Reduced dimension of the caption encoding')
    parser.add_argument('--text', type=str, help='Input text to be converted into image')

    config = parser.parse_args()
    if not os.path.exists(config.save_img):
        os.makedirs('Data' + config.save_img)

    start_time = time.time()
    gen = Generator(batch_size=config.batch_size,
                    img_size=config.img_size,
                    z_dim=config.z_dim,
                    text_embed_dim=config.text_embed_dim,
                    text_reduced_dim=config.text_reduced_dim)

    # Loading the trained model
    G_path = os.path.join(config.final_model, '{}-G.pth'.format('final'))
    gen.load_state_dict(torch.load(G_path))
    # torch.load(gen.state_dict(), G_path)
    gen.eval()

    z = Variable(torch.randn(config.batch_size, config.z_dim)).cuda()
    model = skipthoughts.load_model()
    text_embed = skipthoughts.encode(model, config.text)
    output_img = gen(text_embed, z)
    save_image(output_img.cpu(), config.save_img, nrow=1, padding=0)

    print ('Generated image save to {}'.format(config.save_img))
    print ('Time taken for the task : {}'.format(time.time() - start_time))


if __name__ == '__main':
    test()
