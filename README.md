# Deep Metric Learning

### Learn a deep metric which can be used image retrieval , clustering.
============================

## Pytorch Code for deep metric methods:

- ["Lifted Structure Loss"](
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Song_Deep_Metric_Learning_CVPR_2016_paper.pdf)

        -wait to be done in future

- Contrasstive Loss

- Batch-All-Loss and Batch-Hard-Loss in ["In Defense of Triplet Loss in ReID"](https://arxiv.org/abs/1703.07737)


- HistogramLoss :  ["Learning Deep Embeddings with Histogram Loss"](https://arxiv.org/abs/1611.00822)

- BinDevianceLoss : baseline method in BIER(Deep Metric Learning with BIER: Boosting Independent Embeddings Robustly)

- DistWeightDevianceLoss (my implement of the sampling way in <<sampling matters in deep embedding learning >> combined with BinDevianceLoss)

  I think my implement is better than the sampling way in the paper.

-  ##### KNNSoftmax (ONCA LOSS)

<<Learning a Nonlinear Embedding by Preserving Class Neighbourhood Structur>> Ruslan Salakhutdinov and Geoffrey Hinton

  Though the method is more than 10 years old, It has best performance.

  (R@1 is higher 0.61 on  CUB without test augment with Dim 512 finetuned on pretrained inception-v2)

- And I have a lot of "wrong" ideas during research the DML problems,
I keep them here without description.
You can see the code by yourself, the code is clear and easy for understanding.
If you have any question about losses that  not been mentioned above,
Feel free to ask me.


## Dataset
- [Car-196](http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz)

   first 98 classes as train set and last 98 classes as test set
- [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz)

  first 100 classes as train set and last 100 classes as test set

- [Stanford-Online](ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip)
  
  for the experiments, we split 59,551 images of 11,318 classes for training and 60,502 images of 11,316 classes for testing

  After downloading all the three data file, you should precess them as above, and put the directionary named DataSet in the project.
  We provide a script to precess CUB( Deep_Metric/DataSet/split_dataset.py ). The other two are similar, you can modify the script by yourself.


## Pretrained models in Pytorch

Inceptionn BN network as other metric learning papers do
The download site(http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-239d2248.pth)

~~(to save your time, we already download them down and put on my Baidu YunPan.We also put inception v3 in the Baidu YunPan, the performance of inception v-3 is a little worse(about 1.5% on recall@1 ) than inception BN on CUB/Car datasets.)~~
## Prerequisites

- Computer with Linux or OSX
- For training, an NVIDIA GPU is strongly recommended for speed. CPU is supported but training may be slow.

The pre-trained model inception-v2 is transferred from Caffe, it only can be worked on specific version of Pytorch or Python,
I do not figure out why, and do not which version is best, but if you want to get similar persormance as me
Please create a env as follows:

- Python : 3.5.2 
- [PyTorch](http://pytorch.org)  : (0.2.03)


## Reproducing Car-196 (or CUB-200-2011) experiments

**With our loss based on fussy clustering:**

```bash
sh run_train_00.sh
```

To reproduce other experiments, you can edit the run_train.sh file by yourself.



Future work: I will make the code more clear before 2018 - 4 - 15 . and also share my experiment results. 
