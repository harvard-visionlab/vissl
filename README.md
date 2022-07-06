# vissl

unoffocial torchub for [vissl](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md) models.

VISSL has a large number of potentially useful pre-trained models, but I was unable to successfully load these models because I couldn't crack the davinci code (configuration file pain).

So I made this little interface that let's me load their pre-trained models using torch hub. Please see original source (https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md) for references.

```
  import torch

  model, transform = torch.hub.load("harvard-visionlab/vissl", "alexnet_rotnet_in1k")
```

## jigsaw

-   [ ] resnet50_jigsaw_100perm_in1k
