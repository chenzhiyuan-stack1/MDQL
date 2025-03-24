## A Generalizable and Expressive Meta-Diffusion Policy for RTC Bandwidth Prediction &mdash; Official PyTorch Implementation

## Experiments

### Requirements
Installations of [PyTorch](https://pytorch.org/).

### Training
Running experiments based our code could be quite easy. 
For reproducing the optimal results, we recommend running the following command:
```.bash
python Meta-main-iql-id-gru-TD7_onlyzs_good33.py --env_name meta-v14-id-gru-eta10-TD7-onlyzs-good33 --device 0 --ms online --lr_decay
```

Hyperparameters have been hard coded in `Meta-main-iql-id-gru-TD7_onlyzs_good33.py` for easily reproducing our reported results. 
Definitely, there could exist better hyperparameter settings. Feel free to have your own modifications. 
