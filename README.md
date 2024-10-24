
---

<div align="center">    
 
# DeNetDM (NeurIPS 2024)     

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2024-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://arxiv.org/abs/2403.19863)
 
Conference   
-->   
</div>
 
## Description   


## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/silpavs/denetdm  
cd denetdm
conda env create -f denetdm.yml

 ```   
## Generate datasets
```
python make_dataset.py with server_user make_target=colored_mnist
python make_dataset.py with server_user make_target=cifar10_type0
```
 Next, navigate to any file and run it.   
 ```bash
# run denetdm for any of the datasets 
bash scripts/$DATASET.sh
```
where $DATASET should be replaced by the dataset you wish to run.



## Evaluation
```
python train.py with server_user colored_mnist skewed3 severity4
```

## Updates
- __[07/2023]__ Check out the [Oral talk video](https://www.youtube.com/watch?v=WiSrCWAAUNI) (10 mins) for our ICML paper.
- __[05/2023]__ Paper accepted to [ICML 2023](https://icml.cc/Conferences/2023).
- __[02/2023]__ [arXiv version](https://arxiv.org/abs/2302.12254) posted. Code is released.


## Acknowledgements
This code is partly based on the open-source implementations from [Learning from Failure](https://github.com/alinlab/LfF) and [DFA](https://github.com/kakaoenterprise/Learning-Debiased-Disentangled).


## Citation
If you find this code or idea useful, please cite our work:
```bib
@inproceedings{yang2023change,
  title={Change is Hard: A Closer Look at Subpopulation Shift},
  author={Yang, Yuzhe and Zhang, Haoran and Katabi, Dina and Ghassemi, Marzyeh},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```


## Contact
If you have any questions, feel free to contact us through email (yuzhe@mit.edu & haoranz@mit.edu) or GitHub issues. Enjoy
