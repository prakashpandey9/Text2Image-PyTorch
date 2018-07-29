import os
import torch
import skipthoughts
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import Dataset

# Each batch will have 3 things : true image, its captions(5), and false image(real image but image
# corresponding to an incorrect caption).
# Discriminator is trained in such a way that true_img + caption corresponds to a real example and
# false_img + caption corresponds to a fake example.


class Text2ImageDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.load_flower_dataset()

    def load_flower_dataset(self):
        # It will return two things : a list of image file names, a dictionary of 5 captions per image
        # with image file name as the key of the dictionary and 5 values(captions) for each key.

        print ("------------------  Loading images  ------------------")
        self.img_files = []
        for f in os.listdir(os.path.join(self.data_dir, 'flowers')):
            self.img_files.append(f)

        print ('Total number of images : {}'.format(len(self.img_files)))

        print ("------------------  Loading captions  ----------------")
        self.img_captions = {}
        for class_dir in tqdm(os.listdir(os.path.join(self.data_dir, 'text_c10'))):
            if not 't7' in class_dir:
                for cap_file in class_dir:
                    if 'txt' in cap_file:
                        with open(cap_file) as f:
                            captions = f.read().split('\n')
                        img_file = cap_file[:11] + '.jpg'
                        # 5 captions per image
                        self.img_captions[img_file] = captions[:5]

        print ("---------------  Loading Skip-thought Model  ---------------")
        model = skipthoughts.load_model()
        self.encoded_captions = {}

        print ("------------  Encoding of image captions STARTED  ------------")
        for img_file in self.img_captions:
            self.encoded_captions[img_file] = skipthoughts.encode(model, self.img_captions[img_file])
            # print (type(self.encoded_captions[img_file]))
            # convert it to torch tensor if it is a numpy array

        print ("-------------  Encoding of image captions DONE  -------------")

    def read_image(self, image_file_name):
        image = Image.open(os.path.join(self.data_dir, 'flowers/' + image_file_name))
        # check its shape and reshape it to (64, 64, 3)
        return image

    def get_false_img(self, index):
        false_img_id = np.random.randint(len(self.img_files))
        if false_img_id != index:
            return self.img_files[false_img_id]

        return self.get_false_img(index)

    def __len__(self):

        return len(self.img_files)

    def __getitem__(self, index):

        sample = {}
        sample['true_imgs'] = torch.FloatTensor(self.read_image(self.img_files[index]))
        sample['false_imgs'] = torch.FloatTensor(self.read_image(self.get_false_img(index)))
        sample['true_embed'] = torch.FloatTensor(self.encoded_captions[self.img_files[index]])

        return sample
