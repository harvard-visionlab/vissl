import os
import torch
import torchvision

from models import (
  AlexNetRotNet, AlexnetDeepCluster, VGG16DeepCluster
)

from torchvision.models import resnet50

dependencies = ["torch", "torchvision"]

def _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(resize),
        torchvision.transforms.CenterCrop(crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)    
    ])
    
    return transform

# ======================================================
#  RotNet Models
# ======================================================

def alexnet_rotnet_in1k(pretrained=True, **kwargs):
    model = AlexNetRotNet(**kwargs)
    if pretrained:
      checkpoint_url = "https://visionlab-pretrainedmodels.s3.amazonaws.com/model_zoo/vissl/alexnet_rotnet_model_net_epoch50-af21e82d.pth"
      cache_file_name = "alexnet_rotnet_model_net_epoch50-af21e82d.pth"

      checkpoint = torch.hub.load_state_dict_from_url(
          url=checkpoint_url,
          map_location="cpu",
          file_name=cache_file_name,
          check_hash=True
      )
      state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['network'].items()}
      model.load_state_dict(state_dict, strict=True)
      model.hashid = 'af21e82d'
      model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
      model.weighs_url = checkpoint_url

    transform = _transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return model, transform

def resnet50_rotnet_in1k(pretrained=True, **kwargs):
    model = resnet50(num_classes=4, **kwargs)
    if pretrained:
      checkpoint_url = "https://dl.fbaipublicfiles.com/vissl/model_zoo/rotnet_rn50_in1k_ep105_rotnet_8gpu_resnet_17_07_20.46bada9f/model_final_checkpoint_phase125.torch"
      cache_file_name = "resnet50_rotnet_in1k-5c0b916d.pth"

      checkpoint = torch.hub.load_state_dict_from_url(
          url=checkpoint_url,
          map_location="cpu",
          file_name=cache_file_name,
          check_hash=True
      )
      trunk = checkpoint['classy_state_dict']['base_model']['model']['trunk']
      trunk = {k.replace("_feature_blocks.",""):v for k,v in trunk.items()}
      head = checkpoint['classy_state_dict']['base_model']['model']['heads']
      head = {k.replace("0.clf.0.","fc."):v for k,v in head.items()}
      state_dict = {**trunk, **head}

      model.load_state_dict(state_dict, strict=True)
      model.hashid = '5c0b916d'
      model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
      model.weighs_url = checkpoint_url

    transform = _transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return model, transform    

def resnet50_rotnet_in22k(pretrained=True, **kwargs):
    model = resnet50(num_classes=4, **kwargs)
    if pretrained:
      checkpoint_url = "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_rotnet_in22k_ep105.torch"
      cache_file_name = "resnet50_rotnet_in22k-559277fa.pth"

      checkpoint = torch.hub.load_state_dict_from_url(
          url=checkpoint_url,
          map_location="cpu",
          file_name=cache_file_name,
          check_hash=True
      )
      trunk = checkpoint['classy_state_dict']['base_model']['model']['trunk']
      trunk = {k.replace("_feature_blocks.",""):v for k,v in trunk.items()}
      head = checkpoint['classy_state_dict']['base_model']['model']['heads']
      head = {k.replace("0.clf.0.","fc."):v for k,v in head.items()}
      state_dict = {**trunk, **head}

      model.load_state_dict(state_dict, strict=True)
      model.hashid = '559277fa'
      model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
      model.weighs_url = checkpoint_url

    transform = _transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return model, transform     

# ======================================================
#  Deep Cluster
# ======================================================    

def alexnet_deepcluster_in1k(pretrained=True, **kwargs):
    model = AlexnetDeepCluster(**kwargs)
    if pretrained:
      checkpoint_url = "https://dl.fbaipublicfiles.com/deepcluster/alexnet/checkpoint.pth.tar"
      cache_file_name = "alexnet_deepcluster-3db70837.pth"

      checkpoint = torch.hub.load_state_dict_from_url(
          url=checkpoint_url,
          map_location="cpu",
          file_name=cache_file_name,
          check_hash=True
      )
      state_dict = {k.replace(".module",""):v for k,v in checkpoint['state_dict'].items()}

      model.load_state_dict(state_dict, strict=True)
      model.hashid = '3db70837'
      model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
      model.weighs_url = checkpoint_url

    transform = _transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return model, transform

def vgg16_deepcluster_in1k(pretrained=True, **kwargs):
    model = VGG16DeepCluster(**kwargs)
    if pretrained:
      checkpoint_url = "https://dl.fbaipublicfiles.com/deepcluster/vgg16/checkpoint.pth.tar"
      cache_file_name = "vgg16_deepcluster-b6b90ac1.pth"

      checkpoint = torch.hub.load_state_dict_from_url(
          url=checkpoint_url,
          map_location="cpu",
          file_name=cache_file_name,
          check_hash=True
      )
      state_dict = {k.replace(".module",""):v for k,v in checkpoint['state_dict'].items()}

      model.load_state_dict(state_dict, strict=True)
      model.hashid = 'b6b90ac1'
      model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
      model.weighs_url = checkpoint_url

    transform = _transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return model, transform    