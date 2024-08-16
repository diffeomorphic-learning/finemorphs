# FineMorphs

This repository contains the code and data sets for the paper [FineMorphs: Affine-diffeomorphic sequences for regression](https://arxiv.org/abs/2305.17255) (2023). 

## Requirements

Code requirements are listed in `requirements.txt`.

## Usage

Basic usage of the FineMorphs model is demonstrated in `demos.py`.

## Experiments

The FineMorphs experiments from the paper are presented in list format in `script_experiments.py`, with each experiment a list entry of the form  

```python
experiments += [ ...
```

To run one or more experiments, uncomment the corresponding list entries and execute the script file.   


## Data
Train-test splits in the `data/UCI` directory are from repositories [here](https://github.com/yaringal/DropoutUncertaintyExps/tree/master/UCI_Datasets) and [here](https://github.com/cambridge-mlg/DUN/tree/master/experiments/data/UCI_for_sharing). Data sets in the `data/UCI/airfoil` and `data/UCI/year_prediction_MSD` directories are from their respective UCI repositories, [here](https://archive.ics.uci.edu/ml/datasets/airfoil) and [here](https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd). (Due to size limitations, the latter directory and data set are not contained in this repository and will need to be manually created and downloaded by the user.) 


## Citing

If you find this code useful in your research, please consider citing:

```
@article{lohr2023finemorphs,
  title={FineMorphs: Affine-Diffeomorphic Sequences for Regression},
  author={Lohr, Michele and Younes, Laurent},
  journal={arXiv preprint arXiv:2305.17255},
  year={2023}
}
```

## License

MIT
