"""
Created on Thu Oct 26 11:06:51 2017

Original Author:
Utku Ozbulak - github.com/utkuozbulak

Changes made by:
Thomas Dekelver - git.bdbelux.be/tdekelver
"""
import cv2
import numpy as np
import torch

from misc_functions import get_example_params, save_class_activation_images


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        pass_until = 0
        for module_pos, module in enumerate(self.model[0].modules()):
            if module_pos < pass_until:
                continue # when have a downsampling layer, we take the basic building block instead of each layer separately to avoid tensor shape issues
            elif len([x for x in module.children()]) != 0:
                # Check names of children
                names = [w[0] for w in module.named_modules()]
                if 'downsample' in names:
                    pass_until = module_pos + len(names)
                    #print("downsample \t" + str(module_pos) + '\t pass until \t' + str(pass_until))
                    x = module(x)  # Forward
            else:  # Check to be sure we are not in a Sequential or basisblock
                #print(module_pos)
                x = module(x)  # Forward
                if int(module_pos) == self.target_layer:
                    x.register_hook(self.save_gradient)
                    conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        #x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model[1](x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model[0].zero_grad()
        self.model[1].zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        return cam


if __name__ == '__main__':
    # Get params
    target_example = 2  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)
    # Grad cam
    grad_cam = GradCam(pretrained_model, target_layer=11)
    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, target_class)
    # Save mask
    save_class_activation_images(original_image, cam, file_name_to_export)
    print('Grad cam completed')
