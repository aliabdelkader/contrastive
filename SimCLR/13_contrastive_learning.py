#!/usr/bin/env python
# coding: utf-8

# 
# # Tutorial 13: Self-Supervised Contrastive Learning with SimCLR
# 
# * **Author:** Phillip Lippe
# * **License:** CC BY-SA
# * **Generated:** 2021-10-10T18:35:52.598167
# 
# In this tutorial, we will take a closer look at self-supervised contrastive learning.
# Self-supervised learning, or also sometimes called unsupervised learning, describes the scenario where we have given input data, but no accompanying labels to train in a classical supervised way.
# However, this data still contains a lot of information from which we can learn: how are the images different from each other?
# What patterns are descriptive for certain images?
# Can we cluster the images?
# To get an insight into these questions, we will implement a popular, simple contrastive learning method, SimCLR, and apply it to the STL10 dataset.
# This notebook is part of a lecture series on Deep Learning at the University of Amsterdam.
# The full list of tutorials can be found at https://uvadlc-notebooks.rtfd.io.
# 
# 
# ---
# Open in [![Open In Colab](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHUAAAAUCAYAAACzrHJDAAAIuUlEQVRoQ+1ZaVRURxb+qhdolmbTUVSURpZgmLhHbQVFZIlGQBEXcMvJhKiTEzfigjQg7oNEJ9GMGidnjnNMBs2czIzajksEFRE1xklCTKJiQLRFsUGkoUWw+82pamn79etGYoKek1B/4NW99/tu3e/dquJBAGD27NkHALxKf39WY39gyrOi+i3xqGtUoePJrFmznrmgtModorbTu8YRNZk5cybXTvCtwh7o6NR2KzuZMWNGh6jtVt7nA0ymT5/eJlF9POrh7PAQl6s8bGYa3PUum//htmebVtLRqW0q01M5keTk5FZFzU0oRle3+zxwg5Hgtb+PZiL/ZVohxCI+hL5JgjmfjPxZ26+33BG3dA+ealHPM4gQAo5rU59gsI8bRvl54t3Ca62mvHyUAhtOlLd5WSQpKcluBjumnoCLs1EARkVd9E8l3p9y2i7RbQ1B6pFwu/YDgW8KbHJHMTQrwnjz2oZm9M4pavOCfo5jWrgCaaMVcMs6/pNhDr0+AMN93XlxV7R6DNpyzi7W/OE+yIrsjU6rTrbKV5cd/pNyItOmTbMp6sbBB+EqaYJY4cWE3VUciNt1TpgfcRFv71Fi54xT5kSoyLvOBEJMOMxWXkFlBeBSX4u6Zkcs+3KszYRtiapbNRqF31UgetVuc8z9vBXIv1qD+F1f83B6uDlCUyfsZGepGPpmg01OB7EITQbhS9ribKy+DmP1DUiClLz4bnIHVOqa7BY+Z1wg5g3zgUvyehiNpnJKxSLc/ts76LKm0BzX3c0RNy1yXjDcB5lWoro4iNHQxM+f1kWeWQARAWQS++trISJTp061Kep25X/MycwtjuctSC5rxo7ppi7VNUox5+PhPHtrsS2O1qJ6yx1QujQUzm9sh6hbkBlvvGcN8hYnwjUjH6kjfZEd5c/jitz5Jc5U3ENnFynKl4eB7nyEgP2UZ+Yz3/rVEbyYr27qELrtC4FIC0J7sc7xWnmccdHfRRTs0VB+cA4lt+oFcRR/wUeH8FG5w2Mbx8FQ8TXEvv1xYf4wBP3O2WyL3/UVjpXWgIqaFeUPr+wTmDvUB7njH6/bOv+HRg4SqioAg5GDe1aB3ZeMTJkyRSBqkLsWqSEm0fZVBEN94zEZnYvrdx1JL5cxe+a+AbhSJecRRHW/ikTFRTa38dtQlNZ5CRKwFvUtZU/kvBoEF9Uxni/XqIM+dwKbTw3rhcxIf7gmr2M+H6SMwx8iBzJbw5oxeG3Lv5FX9B3AGaHPS8e8z77H7v9VMpvPG5ug1enh7eGK8h0LBTwUb+GInqzInlRUK65DmTPQu4c3+uQKjwKK77zwUxBX4Tq7yR1RuiwUsqlrABCM6esHdXoy47fk4+prYKy8ZF574x4V5BnHQBuf4g9Z9ld8U36L2aktZNNplNfw7zotwWTy5MkCUft4aLEopJj5/OPHl1BQqeAVOnHgNSQOqmBzq9V9cfEm/yx5ubMGKS9cYPZ3vx2OS/c6PVHUuUO7Y1Pci3BO/1zgq18byebfGemLtNF+6JRtOvMk926ibussZqM+1mNz4TWkH7rCbM5phwGRGDAaoF8fY5OHFnlldAA8sgoEXKnDukA1NgSeNjqkJT9brbN4pC9WRweYXyLugR73c+MYvyWfu0yC6+mjzN1Isfw3FKJS98CU/zI1IHFkFPR52cHL2FJk0sB6kMTERIGo9GzcPkLNfA0cwdwi/hfEYO86ZMd9w+y1egfM2T2Eh/vesMNwljSzuZRT420SW3eqy8N6aHMmwmnFUZ7/PGVPbIoNZvNU1BURdHs0bT2+HjL8sDSM2e6vi4Lj5NW8WOLVA6RTT2azxLV+bglaFNqLieqemS/gWkw7NyoAHo+2dEsiivengjKsPFoqWOvbSh/kxPaxyW/JRzH2Fl3EzD9/xjAefJqB3usKUFn/0Gb+S/d/jy3FN2yLOmnSJJtn6oehByEiHPSeXnDxFGPRnoFoaBJjcdQlbDwcjL1zTNuQpoxD7R0OG0uUTMi0fkVwdzBdYIwcwZunxrVJVLplNm54BZp7jfDfYLoNyqQi1K6KxIdHzmN+QQ2WjFIwUT2zTGdlRXo4NFXVUO4sgX5dFC7f0aP/ZlNeUjFBuL8Xjl6uRuP6aMjSjpjzsH62FDU7JhBuGccEXIvDfJFFBc/gHw80dklfCVYnRaDfpiJcutPA4F7qJsfJeUPQI+1fqMlNhFx1FM0GDqkjFVg7NojlQ0Vt4aM5ReSqcbpaCg8nCW5lRsBvbT4T1TLfFptsfh7gItzuKTdJSEiwKSrt1vcmnEXXrsLbYnWDA1bu+z2WKy9Arq+1KRqdfKsoBo0GcdtEpS/B1bO4v0cFiUhkjskvKcMrWwtAPHuwQq8Z+4LZ1vTQANfXt4J0DwZX9gWa9qh4XDM/voC9JXfwYEMMHJcfNtusn82ihvliVUwg5KrPGVf6GH94ZJpEZBen6EC4qYTHA1dXhW0JIex8txzv//c8lhzXIi/BFxOH9jGbQhZsRalTIBZZ8KkGyZAxeRQvXkFF1TWz/Hm46jNYUnjPbt3JxIkT7f6dSj8qfJJyVvBxgaIlblOyjtysNHWN9fjjqWi7glJfW3/S0Hlj2XnA8PhKT9w6g3Qx3XiXhvuxQsuT1proxBKI/AaZqY1Xz5muvY8G8XkRRCaHsfQsRAFDH/tZPbcYuHotOG0FRIqB4HR3wNVoIPLtz8ycTguu+jpEigE218vd1YCr5m+HpHMvEI9u4LTXwNWaLjl0iPwGAmIpeHx1VeCqTJdPs1/vweweQPO3HC24NhOhnTphwoQnfv6QSY2ICbkNmdSA4h87oaLaiYfn5diIEd4att2erOwJXbPUHp953p6orQVSUVWRAXBT8c/dJ5L9xhzaJGp71GR/wFP8P5V2z10NSC9T93QM2xUg8fHxT+zU9ijeU4naHon8CjFJXFzc8/kn+dN06q9QgF98SYSo2Xen2NjYZy5sR6f+4nLSK5Iam2PH/x87a1YN/t5sBgAAAABJRU5ErkJggg==){height="20px" width="117px"}](https://colab.research.google.com/github/PytorchLightning/lightning-tutorials/blob/publication/.notebooks/course_UvA-DL/13-contrastive-learning.ipynb)
# 
# Give us a ⭐ [on Github](https://www.github.com/PytorchLightning/pytorch-lightning/)
# | Check out [the documentation](https://pytorch-lightning.readthedocs.io/en/latest/)
# | Join us [on Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-pw5v393p-qRaDgEk24~EjiZNBpSQFgQ)

# ## Setup
# This notebook requires some packages besides pytorch-lightning.

# In[1]:


# ! pip install --quiet "torch>=1.6, <1.9" "matplotlib" "pytorch-lightning>=1.3" "seaborn" "torchvision" "torchmetrics>=0.3"


# In[2]:


# ! pip install --quiet "matplotlib" "pytorch-lightning>=1.3" "seaborn" "torchvision" "torchmetrics>=0.3"


# In[3]:


# ! pip install timm==0.4.9


# In[4]:


#get_ipython().system(' pip install opencv-python')


# In[5]:

import os
import urllib.request
from copy import deepcopy
from urllib.error import HTTPError

import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from IPython.display import set_matplotlib_formats
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
from torchvision.datasets import STL10
from tqdm.notebook import tqdm

if __name__ == "__main__":

    plt.set_cmap("cividis")
    # get_ipython().run_line_magic('matplotlib', 'inline')
    set_matplotlib_formats("svg", "pdf")  # For export
    matplotlib.rcParams["lines.linewidth"] = 2.0
    sns.set()

    # Import tensorboard
    # get_ipython().run_line_magic('load_ext', 'tensorboard')

    NUM_WORKERS = 10

    # Setting the seed
    pl.seed_everything(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    print("Number of workers:", NUM_WORKERS)


    # As in many tutorials before, we provide pre-trained models.
    # Note that those models are slightly larger as normal (~100MB overall) since we use the default ResNet-18 architecture.
    # If you are running this notebook locally, make sure to have sufficient disk space available.

    # ## SimCLR
    # 
    # We will start our exploration of contrastive learning by discussing the effect of different data augmentation techniques, and how we can implement an efficient data loader for such.
    # Next, we implement SimCLR with PyTorch Lightning, and finally train it on a large, unlabeled dataset.

    # ### Data Augmentation for Contrastive Learning
    # 
    # To allow efficient training, we need to prepare the data loading such that we sample two different, random augmentations for each image in the batch.
    # The easiest way to do this is by creating a transformation that, when being called, applies a set of data augmentations to an image twice.
    # This is implemented in the class `ContrastiveTransformations` below:

    # In[6]:


    class ContrastiveTransformations:
        def __init__(self, base_transforms, n_views=2):
            self.base_transforms = base_transforms
            self.n_views = n_views

        def __call__(self, x):
            return [self.base_transforms(x) for i in range(self.n_views)]


    # The contrastive learning framework can easily be extended to have more _positive_ examples by sampling more than two augmentations of the same image.
    # However, the most efficient training is usually obtained by using only two.
    # 
    # Next, we can look at the specific augmentations we want to apply.
    # The choice of the data augmentation to use is the most crucial hyperparameter in SimCLR since it directly affects how the latent space is structured, and what patterns might be learned from the data.
    # Let's first take a look at some of the most popular data augmentations (figure credit - [Ting Chen and Geoffrey Hinton](https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html)):
    # 
    # <center width="100%"><img src="https://github.com/PyTorchLightning/lightning-tutorials/raw/main/course_UvA-DL/13-contrastive-learning/simclr_data_augmentations.jpg" width="800px" style="padding-top: 10px; padding-bottom: 10px"></center>
    # 
    # All of them can be used, but it turns out that two augmentations stand out in their importance: crop-and-resize, and color distortion.
    # Interestingly, however, they only lead to strong performance if they have been used together as discussed by [Ting Chen et al. ](https://arxiv.org/abs/2006.10029) in their SimCLR paper.
    # When performing randomly cropping and resizing, we can distinguish between two situations: (a) cropped image A provides a local view of cropped image B, or (b) cropped images C and D show neighboring views of the same image (figure credit - [Ting Chen and Geoffrey Hinton](https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html)).
    # 
    # <center width="100%"><img src="https://github.com/PyTorchLightning/lightning-tutorials/raw/main/course_UvA-DL/13-contrastive-learning/crop_views.svg" width="400px" style="padding-top: 20px; padding-bottom: 0px"></center>
    # 
    # While situation (a) requires the model to learn some sort of scale invariance to make crops A and B similar in latent space, situation (b) is more challenging since the model needs to recognize an object beyond its limited view.
    # However, without color distortion, there is a loophole that the model can exploit, namely that different crops of the same image usually look very similar in color space.
    # Consider the picture of the dog above.
    # Simply from the color of the fur and the green color tone of the background, you can reason that two patches belong to the same image without actually recognizing the dog in the picture.
    # In this case, the model might end up focusing only on the color histograms of the images, and ignore other more generalizable features.
    # If, however, we distort the colors in the two patches randomly and independently of each other, the model cannot rely on this simple feature anymore.
    # Hence, by combining random cropping and color distortions, the model can only match two patches by learning generalizable representations.
    # 
    # Overall, for our experiments, we apply a set of 5 transformations following the original SimCLR setup: random horizontal flip, crop-and-resize, color distortion, random grayscale, and gaussian blur.
    # In comparison to the [original implementation](https://github.com/google-research/simclr), we reduce the effect of the color jitter slightly (0.5 instead of 0.8 for brightness, contrast, and saturation, and 0.1 instead of 0.2 for hue).
    # In our experiments, this setting obtained better performance and was faster and more stable to train.
    # If, for instance, the brightness scale highly varies in a dataset, the
    # original settings can be more beneficial since the model can't rely on
    # this information anymore to distinguish between images.

    # In[7]:


    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


    # In[8]:


    contrast_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=384),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,)),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),

        ]
    )


    # After discussing the data augmentation techniques, we can now focus on the dataset.
    # In this tutorial, we will use the [STL10 dataset](https://cs.stanford.edu/~acoates/stl10/), which, similarly to CIFAR10, contains images of 10 classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck.
    # However, the images have a higher resolution, namely $96\times 96$ pixels, and we are only provided with 500 labeled images per class.
    # Additionally, we have a much larger set of $100,000$ unlabeled images which are similar to the training images but are sampled from a wider range of animals and vehicles.
    # This makes the dataset ideal to showcase the benefits that self-supervised learning offers.
    # 
    # Luckily, the STL10 dataset is provided through torchvision.
    # Keep in mind, however, that since this dataset is relatively large and has a considerably higher resolution than CIFAR10, it requires more disk space (~3GB) and takes a bit of time to download.
    # For our initial discussion of self-supervised learning and SimCLR, we
    # will create two data loaders with our contrastive transformations above:
    # the `unlabeled_data` will be used to train our model via contrastive
    # learning, and `train_data_contrast` will be used as a validation set in
    # contrastive learning.

    # In[9]:


    # unlabeled_data = STL10(
    #     root=DATASET_PATH,
    #     split="unlabeled",
    #     download=True,
    #     transform=ContrastiveTransformations(contrast_transforms, n_views=2),
    # )
    # train_data_contrast = STL10(
    #     root=DATASET_PATH,
    #     split="train",
    #     download=True,
    #     transform=ContrastiveTransformations(contrast_transforms, n_views=2),
    # )


    # In[10]:


    from pathlib import Path


    # In[11]:


    import cv2


    # In[12]:


    dataset_path = "/home/user/SemanticKitti/dataset/sequences/"


    # In[13]:


    from torch.utils.data import Dataset
    from pathlib import Path
    class SemanticKITTI(Dataset):
        def __init__(self, split="train", transforms=None):
            seqs = {
            "train" : [
            '00',
            '02',
            '03',
            '04',
            '05',
            '06',
            '09',
            '10'],
            "val" : [ '07', '01'],
            "test" : [ '08'],
            }
            self.data_paths = []
            for seq in seqs[split]:
                self.data_paths.extend(list((Path(dataset_path) / seq).rglob("**/image_2/*.png")))
            self.data_paths = sorted(self.data_paths)
            self.transforms = transforms
        def __len__(self):
            return len(self.data_paths)

        def __getitem__(self, index):
            img_path = self.data_paths[index]
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return self.transforms(img)




    # In[14]:


    train_dataset = SemanticKITTI("train", ContrastiveTransformations(contrast_transforms, n_views=2))
    val_dataset = SemanticKITTI("val", ContrastiveTransformations(contrast_transforms, n_views=2))


    # Finally, before starting with our implementation of SimCLR, let's look
    # at some example image pairs sampled with our augmentations:

    # We see the wide variety of our data augmentation, including randomly cropping, grayscaling, gaussian blur, and color distortion.
    # Thus, it remains a challenging task for the model to match two, independently augmented patches of the same image.

    # ### SimCLR implementation
    # 
    # Using the data loader pipeline above, we can now implement SimCLR.
    # At each iteration, we get for every image $x$ two differently augmented versions, which we refer to as $\tilde{x}_i$ and $\tilde{x}_j$.
    # Both of these images are encoded into a one-dimensional feature vector, between which we want to maximize similarity which minimizes it to all other images in the batch.
    # The encoder network is split into two parts: a base encoder network $f(\cdot)$, and a projection head $g(\cdot)$.
    # The base network is usually a deep CNN as we have seen in e.g. [Tutorial 5](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html) before, and is responsible for extracting a representation vector from the augmented data examples.
    # In our experiments, we will use the common ResNet-18 architecture as $f(\cdot)$, and refer to the output as $f(\tilde{x}_i)=h_i$.
    # The projection head $g(\cdot)$ maps the representation $h$ into a space where we apply the contrastive loss, i.e., compare similarities between vectors.
    # It is often chosen to be a small MLP with non-linearities, and for simplicity, we follow the original SimCLR paper setup by defining it as a two-layer MLP with ReLU activation in the hidden layer.
    # Note that in the follow-up paper, [SimCLRv2](https://arxiv.org/abs/2006.10029), the authors mention that larger/wider MLPs can boost the performance considerably.
    # This is why we apply an MLP with four times larger hidden dimensions, but deeper MLPs showed to overfit on the given dataset.
    # The general setup is visualized below (figure credit - [Ting Chen et al. ](https://arxiv.org/abs/2006.10029)):
    # 
    # <center width="100%"><img src="https://github.com/PyTorchLightning/lightning-tutorials/raw/main/course_UvA-DL/13-contrastive-learning/simclr_network_setup.svg" width="350px"></center>
    # 
    # After finishing the training with contrastive learning, we will remove the projection head $g(\cdot)$, and use $f(\cdot)$ as a pretrained feature extractor.
    # The representations $z$ that come out of the projection head $g(\cdot)$ have been shown to perform worse than those of the base network $f(\cdot)$ when finetuning the network for a new task.
    # This is likely because the representations $z$ are trained to become invariant to many features like the color that can be important for downstream tasks.
    # Thus, $g(\cdot)$ is only needed for the contrastive learning stage.
    # 
    # Now that the architecture is described, let's take a closer look at how we train the model.
    # As mentioned before, we want to maximize the similarity between the representations of the two augmented versions of the same image, i.e., $z_i$ and $z_j$ in the figure above, while minimizing it to all other examples in the batch.
    # SimCLR thereby applies the InfoNCE loss, originally proposed by [Aaron van den Oord et al. ](https://arxiv.org/abs/1807.03748) for contrastive learning.
    # In short, the InfoNCE loss compares the similarity of $z_i$ and $z_j$ to the similarity of $z_i$ to any other representation in the batch by performing a softmax over the similarity values.
    # The loss can be formally written as:
    # $$
    # \ell_{i,j}=-\log \frac{\exp(\text{sim}(z_i,z_j)/\tau)}{\sum_{k=1}^{2N}\mathbb{1}_{[k\neq i]}\exp(\text{sim}(z_i,z_k)/\tau)}=-\text{sim}(z_i,z_j)/\tau+\log\left[\sum_{k=1}^{2N}\mathbb{1}_{[k\neq i]}\exp(\text{sim}(z_i,z_k)/\tau)\right]
    # $$
    # The function $\text{sim}$ is a similarity metric, and the hyperparameter $\tau$ is called temperature determining how peaked the distribution is.
    # Since many similarity metrics are bounded, the temperature parameter allows us to balance the influence of many dissimilar image patches versus one similar patch.
    # The similarity metric that is used in SimCLR is cosine similarity, as defined below:
    # $$
    # \text{sim}(z_i,z_j) = \frac{z_i^\top \cdot z_j}{||z_i||\cdot||z_j||}
    # $$
    # The maximum cosine similarity possible is $1$, while the minimum is $-1$.
    # In general, we will see that the features of two different images will converge to a cosine similarity around zero since the minimum, $-1$, would require $z_i$ and $z_j$ to be in the exact opposite direction in all feature dimensions, which does not allow for great flexibility.
    # 
    # Finally, now that we have discussed all details, let's implement SimCLR below as a PyTorch Lightning module:

    # In[15]:


    from typing import Dict


    # In[16]:


    import timm


    # In[17]:


    from timm.models.helpers import overlay_external_default_cfg
    from timm.models.vision_transformer import VisionTransformer, default_cfgs, build_model_with_cfg, checkpoint_filter_fn
    from timm.models.registry import register_model


    # In[18]:


    class Image2DTransformer(VisionTransformer):
        def __init__(self, remove_tokens_outputs=False, **kwargs):
          super(Image2DTransformer, self).__init__(**kwargs)
          self.remove_tokens_outputs = remove_tokens_outputs

        def forward_blocks(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            copied from timm Library

            function runs input image and returns outputs of transformer blocks 

            Args:
                x: tensor of input image shape (B, C, H, W)

            Returns
                dictionary map block number -> output of the block
            """
            x = self.patch_embed(x)
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            if self.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = self.pos_drop(x + self.pos_embed)
            outputs = dict()
            for i, block in enumerate(self.blocks):
                x = block(x)
                if self.remove_tokens_outputs:
                    if self.dist_token is None:
                        outputs[str(i)] = x[:, 1:, :] # remove class token output
                    else:
                        outputs[str(i)] = x[:, 2:, :] # remove class and dist tokens outputs
                else:
                    outputs[str(i)] = x
            return outputs


    def _create_transformer_2d(variant, pretrained=False, default_cfg=None, **kwargs):
        """ copied from timm library """
        if default_cfg is None:
            default_cfg = deepcopy(default_cfgs[variant])
        overlay_external_default_cfg(default_cfg, kwargs)
        default_num_classes = default_cfg['num_classes']
        default_img_size = default_cfg['input_size'][-2:]

        num_classes = kwargs.pop('num_classes', default_num_classes)
        img_size = kwargs.pop('img_size', default_img_size)
        repr_size = kwargs.pop('representation_size', None)
        if repr_size is not None and num_classes != default_num_classes:
            # Remove representation layer if fine-tuning. This may not always be the desired action,
            # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
            print("Removing representation layer for fine-tuning.")
            repr_size = None

        if kwargs.get('features_only', None):
            raise RuntimeError('features_only not implemented for Vision Transformer models.')

        model = build_model_with_cfg(
            Image2DTransformer, variant, pretrained,
            default_cfg=default_cfg,
            img_size=img_size,
            num_classes=num_classes,
            representation_size=repr_size,
            pretrained_filter_fn=checkpoint_filter_fn,
            **kwargs)

        return model

    @register_model
    def image_2d_transformer(pretrained=False, **kwargs):
        """
        modified copy from timm 
        DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
        ImageNet-1k weights from https://github.com/facebookresearch/deit.
        """
        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
        model = _create_transformer_2d('vit_deit_base_patch16_384', pretrained=pretrained, **model_kwargs)
        return model

    @register_model
    def image_2d_distilled_transformer(pretrained=False, **kwargs):
        """ 
        modified copy from timm 
        DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
        ImageNet-1k weights from https://github.com/facebookresearch/deit.
        """
        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
        model = _create_transformer_2d(
            'vit_deit_base_distilled_patch16_384', pretrained=pretrained, distilled=True, **model_kwargs)
        return model


    # In[19]:


    class SimCLR(pl.LightningModule):
        def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
            super().__init__()
            self.save_hyperparameters()
            assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
            # Base model f(.)
            self.backbone = timm.create_model("image_2d_distilled_transformer", pretrained=True, remove_tokens_outputs=True)
            self.backbone.reset_classifier(0, '')
            self.fc = nn.Sequential( nn.Flatten(), nn.Linear(576*768, 512), nn.ReLU(inplace=True) )
            # self.convnet = torchvision.models.resnet18(
            #     pretrained=False, num_classes=4 * hidden_dim
            # )  # num_classes is the output size of the last linear layer
            # # The MLP for g(.) consists of Linear->ReLU->Linear
            # self.convnet.fc = nn.Sequential(
            #     self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            #     nn.ReLU(inplace=True),
            #     nn.Linear(4 * hidden_dim, hidden_dim),
            # )

        def configure_optimizers(self):
            optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
            )
            return [optimizer], [lr_scheduler]

        def info_nce_loss(self, batch, mode="train"):
            imgs, _ = batch
            imgs = imgs.type_as(imgs, device=self.device)
            # imgs = torch.cat(imgs, dim=0)

            # Encode all images
            feats = self.backbone.forward_blocks(imgs)["11"]
            feats = torch.flatten(feats, start_dim=1)
            feats = self.fc(feats)
            # print(feats.shape)
            # Calculate cosine similarity
            cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
            # Mask out cosine similarity to itself
            self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
            cos_sim.masked_fill_(self_mask, -9e15)
            # Find positive example -> batch_size//2 away from the original example
            pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
            # InfoNCE loss
            cos_sim = cos_sim / self.hparams.temperature
            nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
            nll = nll.mean()

            # Logging loss
            self.log(mode + "_loss", nll)
            # Get ranking position of positive example
            comb_sim = torch.cat(
                [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
                dim=-1,
            )
            sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
            # Logging ranking metrics
            self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
            self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
            self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

            return nll

        def training_step(self, batch, batch_idx):
            return self.info_nce_loss(batch, mode="train")

        def validation_step(self, batch, batch_idx):
            self.info_nce_loss(batch, mode="val")


    # Alternatively to performing the validation on the contrastive learning loss as well, we could also take a simple, small downstream task, and track the performance of the base network $f(\cdot)$ on that.
    # However, in this tutorial, we will restrict ourselves to the STL10
    # dataset where we use the task of image classification on STL10 as our
    # test task.

    # ### Training
    # 
    # Now that we have implemented SimCLR and the data loading pipeline, we are ready to train the model.
    # We will use the same training function setup as usual.
    # For saving the best model checkpoint, we track the metric `val_acc_top5`, which describes how often the correct image patch is within the top-5 most similar examples in the batch.
    # This is usually less noisy than the top-1 metric, making it a better metric to choose the best model from.

    # In[20]:


    CHECKPOINT_PATH = "/home/user/logs/"


    # In[21]:


    def train_simclr(batch_size, max_epochs=500, **kwargs):
        trainer = pl.Trainer(
            default_root_dir=os.path.join(CHECKPOINT_PATH, "SimCLR"),
            gpus=2,
            max_epochs=max_epochs,
            callbacks=[
                ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc_top5"),
                LearningRateMonitor("epoch"),
            ],
            progress_bar_refresh_rate=1,
        )
        trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

        # Check whether pretrained model exists. If yes, load it and skip training
        pretrained_filename = os.path.join(CHECKPOINT_PATH, "SimCLR.ckpt")
        if os.path.isfile(pretrained_filename):
            print(f"Found pretrained model at {pretrained_filename}, loading...")
            # Automatically loads the model with the saved hyperparameters
            model = SimCLR.load_from_checkpoint(pretrained_filename)
        else:
            train_loader = data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
                num_workers=NUM_WORKERS,
            )
            val_loader = data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                num_workers=NUM_WORKERS,
            )
            pl.seed_everything(42)  # To be reproducable
            model = SimCLR(max_epochs=max_epochs, **kwargs)
            trainer.fit(model, train_loader, val_loader)
            # Load best checkpoint after training
            model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        return model


    # A common observation in contrastive learning is that the larger the batch size, the better the models perform.
    # A larger batch size allows us to compare each image to more negative examples, leading to overall smoother loss gradients.
    # However, in our case, we experienced that a batch size of 256 was sufficient to get good results.

    # In[ ]:


    simclr_model = train_simclr(
        batch_size=10, hidden_dim=128, lr=5e-4, temperature=0.07, weight_decay=1e-4, max_epochs=100
    )


# To get an intuition of how training with contrastive learning behaves, we can take a look at the TensorBoard below:

# In[ ]:


# %tensorboard --logdir ../saved_models/tutorial17/tensorboards/SimCLR/


# ## Logistic Regression
# 
# <div class="center-wrapper"><div class="video-wrapper"><iframe src="https://www.youtube.com/embed/o3FktysLLd4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div></div>
# After we have trained our model via contrastive learning, we can deploy it on downstream tasks and see how well it performs with little data.
# A common setup, which also verifies whether the model has learned generalized representations, is to perform Logistic Regression on the features.
# In other words, we learn a single, linear layer that maps the representations to a class prediction.
# Since the base network $f(\cdot)$ is not changed during the training process, the model can only perform well if the representations of $h$ describe all features that might be necessary for the task.
# Further, we do not have to worry too much about overfitting since we have very few parameters that are trained.
# Hence, we might expect that the model can perform well even with very little data.
# 
# First, let's implement a simple Logistic Regression setup for which we assume that the images already have been encoded in their feature vectors.
# If very little data is available, it might be beneficial to dynamically encode the images during training so that we can also apply data augmentations.
# However, the way we implement it here is much more efficient and can be trained within a few seconds.
# Further, using data augmentations did not show any significant gain in this simple setup.

# class LogisticRegression(pl.LightningModule):
#     def __init__(self, feature_dim, num_classes, lr, weight_decay, max_epochs=100):
#         super().__init__()
#         self.save_hyperparameters()
#         # Mapping from representation h to classes
#         self.model = nn.Linear(feature_dim, num_classes)
# 
#     def configure_optimizers(self):
#         optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
#         lr_scheduler = optim.lr_scheduler.MultiStepLR(
#             optimizer, milestones=[int(self.hparams.max_epochs * 0.6), int(self.hparams.max_epochs * 0.8)], gamma=0.1
#         )
#         return [optimizer], [lr_scheduler]
# 
#     def _calculate_loss(self, batch, mode="train"):
#         feats, labels = batch
#         preds = self.model(feats)
#         loss = F.cross_entropy(preds, labels)
#         acc = (preds.argmax(dim=-1) == labels).float().mean()
# 
#         self.log(mode + "_loss", loss)
#         self.log(mode + "_acc", acc)
#         return loss
# 
#     def training_step(self, batch, batch_idx):
#         return self._calculate_loss(batch, mode="train")
# 
#     def validation_step(self, batch, batch_idx):
#         self._calculate_loss(batch, mode="val")
# 
#     def test_step(self, batch, batch_idx):
#         self._calculate_loss(batch, mode="test")

# The data we use is the training and test set of STL10.
# The training contains 500 images per class, while the test set has 800 images per class.

# img_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# 
# train_img_data = STL10(root=DATASET_PATH, split="train", download=True, transform=img_transforms)
# test_img_data = STL10(root=DATASET_PATH, split="test", download=True, transform=img_transforms)
# 
# print("Number of training examples:", len(train_img_data))
# print("Number of test examples:", len(test_img_data))

# Next, we implement a small function to encode all images in our datasets.
# The output representations are then used as inputs to the Logistic Regression model.

# @torch.no_grad()
# def prepare_data_features(model, dataset):
#     # Prepare model
#     network = deepcopy(model.convnet)
#     network.fc = nn.Identity()  # Removing projection head g(.)
#     network.eval()
#     network.to(device)
# 
#     # Encode all images
#     data_loader = data.DataLoader(dataset, batch_size=64, num_workers=NUM_WORKERS, shuffle=False, drop_last=False)
#     feats, labels = [], []
#     for batch_imgs, batch_labels in tqdm(data_loader):
#         batch_imgs = batch_imgs.to(device)
#         batch_feats = network(batch_imgs)
#         feats.append(batch_feats.detach().cpu())
#         labels.append(batch_labels)
# 
#     feats = torch.cat(feats, dim=0)
#     labels = torch.cat(labels, dim=0)
# 
#     # Sort images by labels
#     labels, idxs = labels.sort()
#     feats = feats[idxs]
# 
#     return data.TensorDataset(feats, labels)

# Let's apply the function to both training and test set below.

# train_feats_simclr = prepare_data_features(simclr_model, train_img_data)
# test_feats_simclr = prepare_data_features(simclr_model, test_img_data)

# Finally, we can write a training function as usual.
# We evaluate the model on the test set every 10 epochs to allow early
# stopping, but the low frequency of the validation ensures that we do not
# overfit too much on the test set.

# def train_logreg(batch_size, train_feats_data, test_feats_data, model_suffix, max_epochs=100, **kwargs):
#     trainer = pl.Trainer(
#         default_root_dir=os.path.join(CHECKPOINT_PATH, "LogisticRegression"),
#         gpus=1 if str(device) == "cuda:0" else 0,
#         max_epochs=max_epochs,
#         callbacks=[
#             ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
#             LearningRateMonitor("epoch"),
#         ],
#         progress_bar_refresh_rate=0,
#         check_val_every_n_epoch=10,
#     )
#     trainer.logger._default_hp_metric = None
# 
#     # Data loaders
#     train_loader = data.DataLoader(
#         train_feats_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=0
#     )
#     test_loader = data.DataLoader(
#         test_feats_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=0
#     )
# 
#     # Check whether pretrained model exists. If yes, load it and skip training
#     pretrained_filename = os.path.join(CHECKPOINT_PATH, f"LogisticRegression_{model_suffix}.ckpt")
#     if os.path.isfile(pretrained_filename):
#         print(f"Found pretrained model at {pretrained_filename}, loading...")
#         model = LogisticRegression.load_from_checkpoint(pretrained_filename)
#     else:
#         pl.seed_everything(42)  # To be reproducable
#         model = LogisticRegression(**kwargs)
#         trainer.fit(model, train_loader, test_loader)
#         model = LogisticRegression.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
# 
#     # Test best model on train and validation set
#     train_result = trainer.test(model, test_dataloaders=train_loader, verbose=False)
#     test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
#     result = {"train": train_result[0]["test_acc"], "test": test_result[0]["test_acc"]}
# 
#     return model, result

# Despite the training dataset of STL10 already only having 500 labeled images per class, we will perform experiments with even smaller datasets.
# Specifically, we train a Logistic Regression model for datasets with only 10, 20, 50, 100, 200, and all 500 examples per class.
# This gives us an intuition on how well the representations learned by contrastive learning can be transfered to a image recognition task like this classification.
# First, let's define a function to create the intended sub-datasets from the full training set:

# def get_smaller_dataset(original_dataset, num_imgs_per_label):
#     new_dataset = data.TensorDataset(
#         *(t.unflatten(0, (10, 500))[:, :num_imgs_per_label].flatten(0, 1) for t in original_dataset.tensors)
#     )
#     return new_dataset

# Next, let's run all models.
# Despite us training 6 models, this cell could be run within a minute or two without the pretrained models.

# results = {}
# for num_imgs_per_label in [10, 20, 50, 100, 200, 500]:
#     sub_train_set = get_smaller_dataset(train_feats_simclr, num_imgs_per_label)
#     _, small_set_results = train_logreg(
#         batch_size=64,
#         train_feats_data=sub_train_set,
#         test_feats_data=test_feats_simclr,
#         model_suffix=num_imgs_per_label,
#         feature_dim=train_feats_simclr.tensors[0].shape[1],
#         num_classes=10,
#         lr=1e-3,
#         weight_decay=1e-3,
#     )
#     results[num_imgs_per_label] = small_set_results

# Finally, let's plot the results.

# dataset_sizes = sorted(k for k in results)
# test_scores = [results[k]["test"] for k in dataset_sizes]
# 
# fig = plt.figure(figsize=(6, 4))
# plt.plot(
#     dataset_sizes,
#     test_scores,
#     "--",
#     color="#000",
#     marker="*",
#     markeredgecolor="#000",
#     markerfacecolor="y",
#     markersize=16,
# )
# plt.xscale("log")
# plt.xticks(dataset_sizes, labels=dataset_sizes)
# plt.title("STL10 classification over dataset size", fontsize=14)
# plt.xlabel("Number of images per class")
# plt.ylabel("Test accuracy")
# plt.minorticks_off()
# plt.show()
# 
# for k, score in zip(dataset_sizes, test_scores):
#     print(f"Test accuracy for {k:3d} images per label: {100*score:4.2f}%")

# As one would expect, the classification performance improves the more data we have.
# However, with only 10 images per class, we can already classify more than 60% of the images correctly.
# This is quite impressive, considering that the images are also higher dimensional than e.g. CIFAR10.
# With the full dataset, we achieve an accuracy of 81%.
# The increase between 50 to 500 images per class might suggest a linear increase in performance with an exponentially larger dataset.
# However, with even more data, we could also finetune $f(\cdot)$ in the training process, allowing for the representations to adapt more to the specific classification task given.
# 
# To set the results above into perspective, we will train the base
# network, a ResNet-18, on the classification task from scratch.

# ## Baseline
# 
# As a baseline to our results above, we will train a standard ResNet-18 with random initialization on the labeled training set of STL10.
# The results will give us an indication of the advantages that contrastive learning on unlabeled data has compared to using only supervised training.
# The implementation of the model is straightforward since the ResNet
# architecture is provided in the torchvision library.

# class ResNet(pl.LightningModule):
#     def __init__(self, num_classes, lr, weight_decay, max_epochs=100):
#         super().__init__()
#         self.save_hyperparameters()
#         self.model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
# 
#     def configure_optimizers(self):
#         optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
#         lr_scheduler = optim.lr_scheduler.MultiStepLR(
#             optimizer, milestones=[int(self.hparams.max_epochs * 0.7), int(self.hparams.max_epochs * 0.9)], gamma=0.1
#         )
#         return [optimizer], [lr_scheduler]
# 
#     def _calculate_loss(self, batch, mode="train"):
#         imgs, labels = batch
#         preds = self.model(imgs)
#         loss = F.cross_entropy(preds, labels)
#         acc = (preds.argmax(dim=-1) == labels).float().mean()
# 
#         self.log(mode + "_loss", loss)
#         self.log(mode + "_acc", acc)
#         return loss
# 
#     def training_step(self, batch, batch_idx):
#         return self._calculate_loss(batch, mode="train")
# 
#     def validation_step(self, batch, batch_idx):
#         self._calculate_loss(batch, mode="val")
# 
#     def test_step(self, batch, batch_idx):
#         self._calculate_loss(batch, mode="test")

# It is clear that the ResNet easily overfits on the training data since its parameter count is more than 1000 times larger than the dataset size.
# To make the comparison to the contrastive learning models fair, we apply data augmentations similar to the ones we used before: horizontal flip, crop-and-resize, grayscale, and gaussian blur.
# Color distortions as before are not used because the color distribution of an image showed to be an important feature for the classification.
# Hence, we observed no noticeable performance gains when adding color distortions to the set of augmentations.
# Similarly, we restrict the resizing operation before cropping to the max.
# 125% of its original resolution, instead of 1250% as done in SimCLR.
# This is because, for classification, the model needs to recognize the full object, while in contrastive learning, we only want to check whether two patches belong to the same image/object.
# Hence, the chosen augmentations below are overall weaker than in the contrastive learning case.

# train_transforms = transforms.Compose(
#     [
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomResizedCrop(size=96, scale=(0.8, 1.0)),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 0.5)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,)),
#     ]
# )
# 
# train_img_aug_data = STL10(root=DATASET_PATH, split="train", download=True, transform=train_transforms)

# The training function for the ResNet is almost identical to the Logistic Regression setup.
# Note that we allow the ResNet to perform validation every 2 epochs to
# also check whether the model overfits strongly in the first iterations
# or not.

# def train_resnet(batch_size, max_epochs=100, **kwargs):
#     trainer = pl.Trainer(
#         default_root_dir=os.path.join(CHECKPOINT_PATH, "ResNet"),
#         gpus=1 if str(device) == "cuda:0" else 0,
#         max_epochs=max_epochs,
#         callbacks=[
#             ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
#             LearningRateMonitor("epoch"),
#         ],
#         progress_bar_refresh_rate=1,
#         check_val_every_n_epoch=2,
#     )
#     trainer.logger._default_hp_metric = None
# 
#     # Data loaders
#     train_loader = data.DataLoader(
#         train_img_aug_data,
#         batch_size=batch_size,
#         shuffle=True,
#         drop_last=True,
#         pin_memory=True,
#         num_workers=NUM_WORKERS,
#     )
#     test_loader = data.DataLoader(
#         test_img_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=NUM_WORKERS
#     )
# 
#     # Check whether pretrained model exists. If yes, load it and skip training
#     pretrained_filename = os.path.join(CHECKPOINT_PATH, "ResNet.ckpt")
#     if os.path.isfile(pretrained_filename):
#         print("Found pretrained model at %s, loading..." % pretrained_filename)
#         model = ResNet.load_from_checkpoint(pretrained_filename)
#     else:
#         pl.seed_everything(42)  # To be reproducable
#         model = ResNet(**kwargs)
#         trainer.fit(model, train_loader, test_loader)
#         model = ResNet.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
# 
#     # Test best model on validation set
#     train_result = trainer.test(model, test_dataloaders=train_loader, verbose=False)
#     val_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
#     result = {"train": train_result[0]["test_acc"], "test": val_result[0]["test_acc"]}
# 
#     return model, result

# Finally, let's train the model and check its results:

# resnet_model, resnet_result = train_resnet(batch_size=64, num_classes=10, lr=1e-3, weight_decay=2e-4, max_epochs=100)
# print(f"Accuracy on training set: {100*resnet_result['train']:4.2f}%")
# print(f"Accuracy on test set: {100*resnet_result['test']:4.2f}%")

# The ResNet trained from scratch achieves 73.31% on the test set.
# This is almost 8% less than the contrastive learning model, and even slightly less than SimCLR achieves with 1/10 of the data.
# This shows that self-supervised, contrastive learning provides
# considerable performance gains by leveraging large amounts of unlabeled
# data when little labeled data is available.

# ## Conclusion
# 
# In this tutorial, we have discussed self-supervised contrastive learning and implemented SimCLR as an example method.
# We have applied it to the STL10 dataset and showed that it can learn generalizable representations that we can use to train simple classification models.
# With 500 images per label, it achieved an 8% higher accuracy than a similar model solely trained from supervision and performs on par with it when only using a tenth of the labeled data.
# Our experimental results are limited to a single dataset, but recent works such as [Ting Chen et al. ](https://arxiv.org/abs/2006.10029) showed similar trends for larger datasets like ImageNet.
# Besides the discussed hyperparameters, the size of the model seems to be important in contrastive learning as well.
# If a lot of unlabeled data is available, larger models can achieve much stronger results and come close to their supervised baselines.
# Further, there are also approaches for combining contrastive and supervised learning, leading to performance gains beyond supervision (see [Khosla et al.](https://arxiv.org/abs/2004.11362)).
# Moreover, contrastive learning is not the only approach to self-supervised learning that has come up in the last two years and showed great results.
# Other methods include distillation-based methods like [BYOL](https://arxiv.org/abs/2006.07733) and redundancy reduction techniques like [Barlow Twins](https://arxiv.org/abs/2103.03230).
# There is a lot more to explore in the self-supervised domain, and more, impressive steps ahead are to be expected.
# 
# ### References
# 
# [1] Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. (2020).
# A simple framework for contrastive learning of visual representations.
# In International conference on machine learning (pp.
# 1597-1607).
# PMLR.
# ([link](https://arxiv.org/abs/2002.05709))
# 
# [2] Chen, T., Kornblith, S., Swersky, K., Norouzi, M., and Hinton, G. (2020).
# Big self-supervised models are strong semi-supervised learners.
# NeurIPS 2021 ([link](https://arxiv.org/abs/2006.10029)).
# 
# [3] Oord, A. V. D., Li, Y., and Vinyals, O.
# (2018).
# Representation learning with contrastive predictive coding.
# arXiv preprint arXiv:1807.03748.
# ([link](https://arxiv.org/abs/1807.03748))
# 
# [4] Grill, J.B., Strub, F., Altché, F., Tallec, C., Richemond, P.H., Buchatskaya, E., Doersch, C., Pires, B.A., Guo, Z.D., Azar, M.G.
# and Piot, B.
# (2020).
# Bootstrap your own latent: A new approach to self-supervised learning.
# arXiv preprint arXiv:2006.07733.
# ([link](https://arxiv.org/abs/2006.07733))
# 
# [5] Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C. and Krishnan, D. (2020).
# Supervised contrastive learning.
# arXiv preprint arXiv:2004.11362.
# ([link](https://arxiv.org/abs/2004.11362))
# 
# [6] Zbontar, J., Jing, L., Misra, I., LeCun, Y. and Deny, S. (2021).
# Barlow twins: Self-supervised learning via redundancy reduction.
# arXiv preprint arXiv:2103.03230.
# ([link](https://arxiv.org/abs/2103.03230))

# ## Congratulations - Time to Join the Community!
# 
# Congratulations on completing this notebook tutorial! If you enjoyed this and would like to join the Lightning
# movement, you can do so in the following ways!
# 
# ### Star [Lightning](https://github.com/PyTorchLightning/pytorch-lightning) on GitHub
# The easiest way to help our community is just by starring the GitHub repos! This helps raise awareness of the cool
# tools we're building.
# 
# ### Join our [Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-pw5v393p-qRaDgEk24~EjiZNBpSQFgQ)!
# The best way to keep up to date on the latest advancements is to join our community! Make sure to introduce yourself
# and share your interests in `#general` channel
# 
# 
# ### Contributions !
# The best way to contribute to our community is to become a code contributor! At any time you can go to
# [Lightning](https://github.com/PyTorchLightning/pytorch-lightning) or [Bolt](https://github.com/PyTorchLightning/lightning-bolts)
# GitHub Issues page and filter for "good first issue".
# 
# * [Lightning good first issue](https://github.com/PyTorchLightning/pytorch-lightning/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
# * [Bolt good first issue](https://github.com/PyTorchLightning/lightning-bolts/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
# * You can also contribute your own notebooks with useful examples !
# 
# ### Great thanks from the entire Pytorch Lightning Team for your interest !
# 
# ![Pytorch Lightning](data:image/png;base64,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MERFOjYxNjMxNjY4IiBkYXRhLXBqYXgtdHJhbnNpZW50PSJ0cnVlIi8+PG1ldGEgbmFtZT0iaHRtbC1zYWZlLW5vbmNlIiBjb250ZW50PSJlODU4ZjJjYzQ3YTJhNGY1NTFhZjQ3MmVhZDBhMzEwMjFiYTAwYmMyMTk5ZjYyYWE0NDJkMGJiZjFkMDNmYjAxIiBkYXRhLXBqYXgtdHJhbnNpZW50PSJ0cnVlIi8+PG1ldGEgbmFtZT0idmlzaXRvci1wYXlsb2FkIiBjb250ZW50PSJleUp5WldabGNuSmxjaUk2SWlJc0luSmxjWFZsYzNSZmFXUWlPaUl3UXpBMU9qTXdOek02TVRBM09ERTBSam94T1RFNU1FUkZPall4TmpNeE5qWTRJaXdpZG1semFYUnZjbDlwWkNJNklqRXdPREEwT0Rnd09UTTVOelEwTmpneU1EQWlMQ0p5WldkcGIyNWZaV1JuWlNJNkltbGhaQ0lzSW5KbFoybHZibDl5Wlc1a1pYSWlPaUpwWVdRaWZRPT0iIGRhdGEtcGpheC10cmFuc2llbnQ9InRydWUiLz48bWV0YSBuYW1lPSJ2aXNpdG9yLWhtYWMiIGNvbnRlbnQ9IjVlOGU0YzNkZDlmYjVmMDExMWViNTAxMzJlMjJlZjg0MDJkZmZlM2ViNGU2ZGNkOThkNmFkMWNkZDU3MWRjNzgiIGRhdGEtcGpheC10cmFuc2llbnQ9InRydWUiLz7sPWl32ziS3/MruOrt3t3XocT7SGzPcxznmDixJ3bSx8w8LkRCEmOKZPOwrZ7X/32rAFKkKMmkDufonemMSYJAoVComwD06JEA/zuY0owIIZnSw94kuqGJSxJPTPPhJ+pmYkbGPcGNwoyG2WEvoXGU+lmUzJ7IpmUohqlIPcEjGRHjT+ROzBISpj5UPXr06NEi6LGfTfKheE1nw4h1MImSzM2zdCX4x2mUJy4V3cijKzs47GVJDq8GR9hRs7OUBoA89cTAD697wg0JcsrBOxzwOqSbBBlH0TigIiBFRSCNP/JdkvlRWEPala/z5+Ib5dXrjy8MVQ9msZv+dBE91679INLFT8fO7bMXsystf/tL76hJlHbwb670cWpNpNsbQsZnb44//vSbNRzSd7fhr7++kRP557/N0p/Vuzw5+7AN+F9/n3ycnb64HZq3KpXE/Pwq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){height="60px" width="240px"}
