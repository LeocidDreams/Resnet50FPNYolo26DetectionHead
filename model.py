##############################
###Model Configuration########
##############################
#checking  libraries and GPU situtaions
import os
import glob
from pathlib import Path
import sys

from collections import OrderedDict
from typing import List
from types import NoneType

#ignoring warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as p

import torch
import torch.nn as nn

import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models import ResNet50_Weights

from ultralytics import YOLO
from ultralytics.nn.modules.head import OBB26

from ultralytics.utils.loss import E2ELoss



##setting configuration
os.environ['CUDA_LAUNCH_BLOCKING']='1'
os.environ['TORCH_USE_CUDA_DSA']='1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device

class Resnet50Yolo26(nn.Module):
    def __init__(self,yoloWeights:Path|str='yolo26n.pt'):
        ####resnet feature extractor(backbone)
        super(Resnet50Yolo26,self).__init__()

        ###resnet backbone
        resnet=models.resnet50(weights='IMAGENET1K_V2').to(device)
        self.conv1=resnet.conv1
        self.bn1=resnet.bn1 #may not required
        self.relu=resnet.relu #may not required
        self.maxpool=resnet.maxpool #may not required
        self.layer1=resnet.layer1
        self.layer2=resnet.layer2
        self.layer3=resnet.layer3
        self.layer4=resnet.layer4


        ###FPN neck
        in_channel_list=[256, 512, 1024, 2048]
        out_channels=256
        self.fpn=FeaturePyramidNetwork(in_channels_list=in_channel_list,out_channels=out_channels).to(device)

        ###YOLO 26 detection head 
        self._yoloWeights=yoloWeights
        self.detectionHead=''
        #checking yolo weigthed path
        if self._checkingWeights():
            # model=YOLO(self._yoloWeights)
            # ##detection head
            # classify=model.model.model[-1].to(device)
            # self.detectionHead=nn.Sequential(
            #     classify
            # )
            from ultralytics.nn.modules.head import OBB26
            self.detectionHead=OBB26(
                nc=15,
                end2end=False,
                ch=(256,256,256,256)
            )
        if isinstance(self.detectionHead,str) or isinstance(self.detectionHead,NoneType):
            raise FileNotFoundError("Couldn't extract yolo detection head,please looking up current yolo models")


    ##defining helper functions
    #checking weights
    def _checkingWeights(self):
        """ 
            -checking current implemented weights file
        """
        if isinstance(self._yoloWeights,Path):
            #converting to string
            string_=str(self._yoloWeights)

            if string_.endswith('.pt') or string_.endswith('.pth'):
                return True
        string_=str(self._yoloWeights)
        if string_.endswith('.pt') or string_.endswith('.pth'):
            return True

        else:
            raise TypeError('Incomptatible Type,it must be .pt and .pth file format!!!')
            yield False

        


    def forward(self,x):
        
        #resnet50 backbone
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        # x=self.layer1(x)
        # layer1=x.copy()

        # x=self.layer2(x)
        # layer2=x.copy()

        # x=self.layer3(x)
        # layer3=x.copy()
        
        # x=self.layer4(x)
        # layer4=x.copy()
        layer1=self.layer1(x)
        layer2=self.layer2(layer1)
        layer3=self.layer3(layer2)
        layer4=self.layer4(layer3)

        features=OrderedDict(
            {
                'layer1':layer1,
                'layer2':layer2,
                'layer3':layer3,
                'layer4':layer4
            }
        )

        ##Feature pyramid network
        fpn_output=self.fpn(features)
        fpn_features= list(fpn_output.values())
        # fpn_features = fpn_features[:-1]  # 3 feature maps



        # return fpn_features
        # return fpn_output
    
        # return self.detectionHead(
        #    fpn_features
        # )
        return self.detectionHead(
            fpn_features
        )