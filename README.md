# ICNN

This repository is a pytorch implementation of interpretable convolutional neural network
([arXiv](https://arxiv.org/abs/1710.00935), [IEEE T-PAMI](https://ieeexplore.ieee.org/document/9050545)). 

It is created by [Wen Shen](https://ada-shen.github.io), Ping Zhao, Qiming Li, [Chao Li](
http://www.ict.cas.cn/sourcedb_2018_ict_cas/cn/jssrck/201810/t20181030_5151364.html).


## Notice
- **model**

The project now supports three different vggs (vgg_vd_16, vgg_m, vgg_s),
                                         alexnet, resnet-18/50, and densenet-121.
                                         
You can add your own model in the `/model` folder and register the model in `/tools/init_model.py`.

- **dataset**

The project now supports **vocpart, ilsvrc animalpart, cub200, 
                         celeba, voc2010_crop, helen**.
                         
You can add your own dataset in the way similar to the datasets above. 

**Note that** in our code, we will first make the data into imdb file, 
so if your dataset is large, the preprocessing time may be long, 
and the generated imdb file will be relatively large.

## Requirement

The environment should have all packages in [requirements.txt](./requirements.txt)
```bash
$ pip install -r requirements.txt
```

You can see that we recommend **pytorch=1.2.0**, this is because we find some bugs when pytorch=1.4.0,
but there is no such problem in pytorch 1.2.0. We will continue to study this problem.

## Usage
Here, we take **resnet-18 + voc2010_crop bird classification** as an example.

To get the training results, we can run:
```bash
$ python demo.py --model resnet_18 --dataset voc2010_crop --label_name bird
```
After running the instructions above, you will get a new folder whose path is
`/resnet_18/voc2010_crop/bird` in the `/task/classification` folder.

the new folder `bird` will contain a subfolder named `0` (correspond to your task_id) and three mat files (mean.mat, train.mat and val.mat).
the `0` folder stores the model of every 10 epoches and log which contains 
**train/val loss** and **train/val accuracy**  during network training.

You can use the trained model to calculate other metrics or to look at middle-level features.

<!--our experiment environment: 
    python: 3.7.7
    torch: 1.2.0
    torchvision: 0.4.0a0
    cuda: 10.2
    gpu: 2080Ti
-->

## Citation

If you use this project in your research, please cite it.

```
@inproceedings{zhang2018interpretable,
 title={Interpretable convolutional neural networks},
 author={Zhang, Quanshi and Wu, Nianying and Zhu, Song-Chun},
 booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
 pages={8827--8836},
 year={2018}
}
```



