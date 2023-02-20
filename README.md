# Disentangled Multimodal Representation Learning for Recommendation

This is our implementation for the paper:

Fan Liu*, Huilin Chen, Zhiyong Cheng, Anan Liu, Liqiang Nie, Mohan Kankanhalli. [Disentangled Multimodal Representation Learning for Recommendation](https://arxiv.org/pdf/2203.05406.pdf). IEEE Transactions on Multimedia. (“*”= Corresponding author)

**Please cite our paper if you use our codes or datasets. Thanks!**

### Requirements

- CUDA 10.0
- Python 3.7

To run first download a full dataset (see below). For example,

```
$ pip install -r requirements.txt
# Adjust the batch size depending on your GPU memory capabilities
$ python DMRL.py --dataset Clothing  --batch_size=64
```

### Dataset
We provide five processed datasets: Amazon-Office, Amazon-Clothing, Amazon-Baby, Amazon-ToysGames, Amazon-Sports.

All of the above datasets could be downloaded from :
- Google Drive [Link](https://drive.google.com/drive/folders/1EmehilbrTMbW5pV2RIHNhopV_hnupvDj?usp=sharing)


Last Update Date: DEC. 04, 2022
