
---

<div align="center">    
 
# DeNetDM: Debiasing by Network Depth Modulation (NeurIPS 2024)     

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2024-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://arxiv.org/abs/2403.19863)
 
Conference   
-->   
</div>

##  
This repository provides the official PyTorch implementation of the paper [DeNetDM: Debiasing by Network Depth Modulation](https://nips.cc/virtual/2024/poster/96916)

> **Abstract:** 
When neural networks are trained on biased datasets, they tend to inadvertently learn spurious correlations, leading to challenges in achieving strong generalization and robustness. Current approaches to address such biases typically involve utilizing bias annotations,
> reweighting based on pseudo-bias labels, or enhancing diversity within bias-conflicting data points through augmentation techniques. We introduce DeNetDM, a novel debiasing method based on the observation that shallow neural networks prioritize learning core attributes,
> while deeper ones emphasize biases when tasked with acquiring distinct information. Using a training paradigm derived from Product of Experts, we create both biased and debiased branches with deep and shallow architectures and then distill knowledge to produce the target debiased model. 
> Extensive experiments and analyses demonstrate that our approach outperforms current debiasing techniques, achieving a notable improvement of around 5% in three datasets, encompassing both synthetic and real-world data. Remarkably, DeNetDM accomplishes this without requiring
> annotations pertaining to bias labels or bias types, while still delivering performance on par with supervised counterparts. Furthermore, our approach effectively harnesses the diversity of bias-conflicting points within the data, surpassing previous methods and obviating the need 
> for explicit augmentation-based methods to enhance the diversity of such bias-conflicting points.
<p align="center">
  <img src="assets/teaser_diagram.png" />
</p>


## How to run   
First, install dependencies   
```bash
git clone https://github.com/kadarsh22/DeNetDM  # clone project   
cd DeNetDM
conda env create -f denetdm.yml # create conda environment
conda activate denetdm
 ```   
## Generate/Download datasets
Create datasets (coloredmnist, corruptedcifar10) by following instruction given in [Learning from Failure](https://github.com/alinlab/LfF)
BFFHQ dataset can be downloaded from the url given in [DFA](https://github.com/kakaoenterprise/Learning-Debiased-Disentangled)

## Training  
 ```bash
# run denetdm for any of the datasets 
bash scripts/$DATASET.sh
```
where $DATASET should be replaced by the dataset you wish to run (coloredmnist/ corruptedcifar10/ bffhq).

## Pretrained Models
The pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/1lajUDMpj9g0wwS_eFMqNHyVczSVrdwLJ?usp=sharing)

## Evaluation
For evaluating pretrained models, update pretrained model path in config.py of corresponding dataset and set train = False in main.py and then execute 
```
python main.py with server_user colored_mnist skewed1 severity4
python main.py with server_user corrupted_cifar10 skewed1 severity4
python3 main.py with bffhq
```


## Updates

- __[27/09/2024]__ Paper accepted to [NeurIPS 2024](https://nips.cc/virtual/2024/poster/96916).
- __[24/10/2024]__ [arXiv version](https://arxiv.org/abs/2403.19863) posted. Code is released.
- __[12/12/2024]__ In person poster presentation .


## Acknowledgements
This code is partly based on the open-source implementations from [Learning from Failure](https://github.com/alinlab/LfF) and [DFA](https://github.com/kakaoenterprise/Learning-Debiased-Disentangled).


## Citation
If you find this code or idea useful, please cite our work:
```bib
@misc{sreelatha2024denetdmdebiasingnetworkdepth,
      title={DeNetDM: Debiasing by Network Depth Modulation}, 
      author={Silpa Vadakkeeveetil Sreelatha and Adarsh Kappiyath and Abhra Chaudhuri and Anjan Dutta},
      year={2024},
      eprint={2403.19863},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2403.19863}, 
}
```
