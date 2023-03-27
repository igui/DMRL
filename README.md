# Disentangled Multimodal Representation Learning for Recommendation

This is our implementation for the paper:

Fan Liu*, Huilin Chen, Zhiyong Cheng, Anan Liu, Liqiang Nie, Mohan Kankanhalli. [Disentangled Multimodal Representation Learning for Recommendation](https://arxiv.org/pdf/2203.05406.pdf). IEEE Transactions on Multimedia. (“*”= Corresponding author)

**Please cite our paper if you use our codes or datasets. Thanks!**

### Requirements

- [Python 3.7](https://www.python.org/downloads/release/python-379/)
- [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive)
- [cuDNN v7.6.5 for CUDA 10.0](https://developer.nvidia.com/rdp/cudnn-archive)
- [Tensorflow 1.15](https://www.tensorflow.org/versions/r1.15/api_docs/python/tfq)

To run first download a full dataset (see below). For example, download the
Clothing one and store it in Data/Clothing. The files in the Git
repository do not have image or textual features.

For example, download them into $$

```
$ pip install -r requirements.txt
$ python DMRL.py --dataset Clothing
```

This process takes around 5 hours in a GTX 2060 with 6GB of RAM.

### Dataset
We provide five processed datasets: Amazon-Office, Amazon-Clothing, Amazon-Baby, Amazon-ToysGames, Amazon-Sports.

All of the above datasets could be downloaded from :
- Google Drive [Link](https://drive.google.com/drive/folders/1EmehilbrTMbW5pV2RIHNhopV_hnupvDj?usp=sharing)


Last Update Date: DEC. 04, 2022
