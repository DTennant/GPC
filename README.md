# Learning Semi-supervised Gaussian Mixture Models for Generalized Category Discovery

This repo contains code for our paper: [Learning Semi-supervised Gaussian Mixture Models for Generalized Category Discovery](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_Learning_Semi-supervised_Gaussian_Mixture_Models_for_Generalized_Category_Discovery_ICCV_2023_paper.pdf)



## Contents

[:running: 1. Running](#running)


[:clipboard: 2. Citation](#cite)




## <a name="running"/> :running: Running

### Dependencies

```
pip install -r requirements.txt
```

### Config

Set paths to datasets, pre-trained models and desired log directories in ```config.py```.
Also set the experiment paths in ```bash_scripts/run.sh```.


### Datasets

We use fine-grained benchmarks in this paper, including:                                                                                                                    
                                                                                                                                                                  
* [The Semantic Shift Benchmark (SSB)](https://github.com/sgvaze/osr_closed_set_all_you_need#ssb) and [Herbarium19](https://www.kaggle.com/c/herbarium-2019-fgvc6)

We also use generic object recognition datasets, including:

* [CIFAR-10/100](https://pytorch.org/vision/stable/datasets.html) and [ImageNet](https://image-net.org/download.php)

Please follow [this repo](https://github.com/CVMI-Lab/SimGCD) or [this repo](https://github.com/sgvaze/generalized-category-discovery) to set up the data.


### Scripts

**Train representation**:

```
bash bash_scripts/run.sh
```

## <a name="cite"/> :clipboard: Citation

If you use this code in your research, please consider citing our paper:
```
@InProceedings{Zhao_2023_ICCV,
    author    = {Zhao, Bingchen and Wen, Xin and Han, Kai},
    title     = {Learning Semi-supervised Gaussian Mixture Models for Generalized Category Discovery},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {16623-16633}
}
```
## Acknowledgements

The codebase is largely built on this repo: https://github.com/sgvaze/generalized-category-discovery.
