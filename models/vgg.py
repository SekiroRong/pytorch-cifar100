"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
import torch.onnx 
import onnx 
import onnxruntime
import numpy as np

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))

def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))


if __name__ == '__main__':
    weights = '/home/sekiro/pytorch-cifar100/checkpoint/vgg16/Tuesday_14_March_2023_19h_04m_37s/vgg16-20-regular.pth'
    model = vgg16_bn()
    model.load_state_dict(torch.load(weights))
    dummy_input = torch.randn(1, 3, 32, 32) 
    model(dummy_input)
    # with torch.no_grad(): 
    #     torch.onnx.export( 
    #         model, 
    #         dummy_input, 
    #         "vgg.onnx", 
    #         opset_version=11, 
    #         input_names=['input'], 
    #         output_names=['output'])
    
    onnx_model = onnx.load("vgg.onnx") 
    try: 
        onnx.checker.check_model(onnx_model) 
    except Exception: 
        print("Model incorrect") 
    else: 
        print("Model correct")

    dummy_input_np = np.random.rand(1, 3, 32, 32).astype(np.float32)
    ort_session = onnxruntime.InferenceSession("vgg.onnx")
    ort_inputs = {'input': dummy_input_np}
    ort_output = ort_session.run(['output'], ort_inputs)[0]
    print(ort_output)