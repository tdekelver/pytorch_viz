"""
Created on Thu Oct 26 14:19:44 2017

Original Author:
Utku Ozbulak - github.com/utkuozbulak

Changes made by:
Thomas Dekelver - git.bdbelux.be/tdekelver
"""
import os
import cv2
import numpy as np

from torch.optim import SGD
from torchvision import models

from misc_functions import preprocess_image, recreate_image, save_image


class ClassSpecificImageGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent
    """
    def __init__(self, model, target_class,mean,std,lr = 6,min_loss = -1000,name = 'generated', optim = None):
        self.mean = mean # mean of channels (in fastai: data.norm.keywords['mean'])
        self.std = std # std of channels (in fastai: data.norm.keywords['std'])
        self.lr = lr # initial learning rate to use (default 6)
        self.min_loss = min_loss # min loss to obtain before stopping convergence (default -1000)
        self.name = name # name of images to save
        self.model = model
        self.model.eval()
        self.target_class = target_class
        # Generate a random image
        self.created_image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))
        self.optim = optim
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def generate(self):
        initial_learning_rate = self.lr
        loss = 10
        i = 0
        if self.optim == None:
            optimizer = SGD([self.processed_image], lr=initial_learning_rate)
        else:
            optimizer = self.optim
        while loss >= self.min_loss:
            # Process image and return variable
            self.processed_image = preprocess_image(self.created_image, self.mean, self.std, False)

            # Forward
            output = self.model(self.processed_image)
            # Target specific class
            class_loss = -output[0, self.target_class]
            if loss > class_loss.data.numpy():
                loss = class_loss.data.numpy()
            if i % 25 == 0:
                print('Iteration:', str(i), 'Loss', "{0:.2f}".format(class_loss.data.numpy()))
            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image, self.mean, self.std)
            if i % 25 == 0:
                # Save image
                im_path = '../generated/'+ str(self.name)+'_iteration_'+str(i)+'.jpg'
                save_image(self.created_image, im_path)
            i += 1
        im_path = '../generated/'+ str(self.name)+'_iteration_final.jpg'
        save_image(self.created_image,im_path)
        return self.processed_image


if __name__ == '__main__':
    target_class = 130  # Flamingo
    pretrained_model = models.resnet18(pretrained=True)
    csig = ClassSpecificImageGeneration(pretrained_model, target_class,mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    csig.generate()
