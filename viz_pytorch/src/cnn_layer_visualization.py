"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np

import torch
from torch.optim import Adam
from torchvision import models

from misc_functions import preprocess_image, recreate_image, save_image


class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter,mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.mean = mean  # mean of channels (in fastai: data.norm.keywords['mean'])
        self.std = std  # std of channels (in fastai: data.norm.keywords['std'])
        self.conv_output = 0
        # Generate a random image
        self.created_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        # Create the folder to export images if not exists
        self.layers = []
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def create_layer_list(self):
        pass_until = 0
        for module_pos, module in enumerate(self.model.modules()):
            if module_pos < pass_until:
                continue  # when have a downsampling layer, we take the basic building block instead of each layer separately to avoid tensor shape issues
            elif len([x for x in module.children()]) != 0:
                # Check names of children
                names = [w[0] for w in module.named_modules()]
                if 'downsample' in names:
                    pass_until = module_pos + len(names)
                    # print("downsample \t" + str(module_pos) + '\t pass until \t' + str(pass_until))
                    self.layers.append(module)  # Forward
            else:  # Check to be sure we are not in a Sequential or basisblock
                # print(module_pos)
                self.layers.append(module)  # Forward
        return self.layers

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # Hook the selected layer
        self.layers[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self, optim = None, iterations = 30):
        # Hook the selected layer
        self.hook_layer()
        # Process image and return variable
        self.processed_image = preprocess_image(self.created_image, self.mean, self.std, False)
        # Define optimizer for the image
        if optim == None:
            optimizer = Adam([self.processed_image], lr=0.1, weight_decay=1e-6)
        else:
            optimizer = optim

        for i in range(1, iterations + 1):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = self.processed_image
            for index, layer in enumerate(self.layers):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image, self.mean, self.std)
            # Save image
            if i % 5 == 0:
                im_path = '../generated/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)

    def visualise_layer_without_hooks(self, optim = None, iterations = 30):
        # Process image and return variable
        self.processed_image = preprocess_image(self.created_image, self.mean, self.std, False)
        # Define optimizer for the image
        if optim == None:
            optimizer = Adam([self.processed_image], lr=0.1, weight_decay=1e-6)
        else:
            optimizer = optim
        for i in range(1, iterations + 1):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = self.processed_image
            for index, layer in enumerate(self.layers):
                # Forward pass layer by layer
                x = layer(x)
                if index == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
            self.conv_output = x[0, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image, self.mean, self.std)
            # Save image
            if i % 5 == 0:
                im_path = '../generated/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)


if __name__ == '__main__':
    cnn_layer = 17
    filter_pos = 5
    # Fully connected layer is not needed
    pretrained_model = models.resnet32(pretrained=True)[0]
    layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos)

    # Generate layer list
    layer_vis.create_layer_list()

    # Layer visualization with pytorch hooks
    layer_vis.visualise_layer_with_hooks()

    # Layer visualization without pytorch hooks
    # layer_vis.visualise_layer_without_hooks()
