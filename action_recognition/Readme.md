## Action Recognition

We follow the 3D skeleton-based action recognition setup and [implementation](https://github.com/lshiwjx/2s-AGCN) from Shi et al. [2] 

### Task

**Sample** `(n_frames, feat_dim)`: Each action segment (start-end span) from BABEL is divided into contiguous 5-second chunks. See the [paper](https://arxiv.org/pdf/2106.09696.pdf) for more details. 
**Label** `<int>`: Index of the ground-truth action label of the segment that the current chunk belongs to. 


### Features 

We extract the joint positions (in `x, y, z` co-ordinates) from the AMASS mocap sequences in NTU RGB+D [1] skeleton format. There are 25 joints, resulting in `feat_dim=25*3=75`. 

Each sample is a 5-second chunk @ 30fps, resulting in `n_frames=150`. 

Pre-preprocessing of the skeleton joints follows Shi et al. [2]. Download the pre-processed sample features and corresponding labels: 

```
cd data/
wget https://human-movement.is.tue.mpg.de/babel_feats_labels.tar.gz
tar -xzvf babel_feats_labels.tar.gz -C ./
```

### Training and Inference 

Set up and activate a virtual environment:

```
python3 -m venv babel-env
source $PWD/babel-env/bin/activate
$PWD/babel-env/bin/pip install --upgrade pip setuptools
$PWD/babel-env/bin/pip install -r requirements.txt
```

#### Model 

We use [this](https://github.com/lshiwjx/2s-AGCN) implementation for the 2S-AGCN [2] model for 3D skeleton-based action recognition. Note that we use only the Joint-stream alone. 


#### Training

To train a model with CE loss:

From the top directory `babel/`, enter the following to train a model with the Cross-Entropy loss:

```python action_recognition/main.py --config action_recognition/config/babel_v1.0/train_60.yaml```

To train a model with Focal loss [3] with class-balancing [4]: 

```python action_recognition/main_wl.py --config action_recognition/config/babel_v1.0/train_60_wfl.yaml```

You can use the repsective configuration files inside `config/babel_v1.0` to train the model with `120` classes in both ways.


### Inference 

Provide the path to the trained model in the `weights` key in the respective config file. 

To perform inference, use the same command as when training, and pass the test config file as argument. E.g.:

```python action_recognition/main.py --config action_recognition/config/babel_v1.0/test_60.yaml```

or

```python action_recognition/main_wl.py --config action_recognition/config/babel_v1.0/test_60_wfl.yaml```

To save the predicted scores to disk, in the config file, set `save_score: True`. 

### Pre-trained models 

Download the checkpoints from the links below and place them in `action_recognition/ckpts/`. 

Performing inference on the validation set should result in the following performance. 

| \# Classes | Loss type  | Ckpt  | Top-5 | Top-1 | Top-1-norm | 
|---|---|---|---|---|--|
| BABEL-60 | CE | [ntu_sk_60_agcn_joint_const_lr_1e-3-17-6390.pt](https://human-movement.is.tue.mpg.de/release/ckpts/ntu_sk_60_agcn_joint_const_lr_1e-3-17-6390.pt) | 0.74 | 0.42 | 0.24 | 
| BABEL-60 | Focal | [wfl_ntu_sk_60_agcn_joint_const_lr_1e-3-93-33370.pt](https://human-movement.is.tue.mpg.de/release/ckpts/wfl_ntu_sk_60_agcn_joint_const_lr_1e-3-93-33370.pt) | 0.69 | 0.34 | 0.30 | 
| BABEL-120 | CE | [ntu_sk_120_agcn_joint_const_lr_1e-3-15-12240.pt](https://human-movement.is.tue.mpg.de/release/ckpts/ntu_sk_120_agcn_joint_const_lr_1e-3-15-12240.pt) | 0.72 | 0.4 | 0.16 | 
| BABEL-120 | Focal | [wfl_ntu_sk_120_agcn_joint_const_lr_1e-3-157-60356.pt](https://human-movement.is.tue.mpg.de/release/ckpts/wfl_ntu_sk_120_agcn_joint_const_lr_1e-3-157-60356.pt) | 0.59 | 0.29 | 0.23 |

**Note:** The models are *only* trained with dense labels from `train.json` (See [project webpage](https://babel.is.tue.mpg.de/data.html) for more details about the data). 


### Challenge 

Coming soon!


### Metrics 

**Description**

1. **Top-1** measures the accuracy of the highest-scoring prediction. 
2. **Top-5** evaluates whether the ground-truth category is present among the top 5 highest-scoring predictions. 
    1. It accounts for labeling noise and inherent label ambiguity. 
    2. It also accounts for the possible association of multiple action categories with a single input movement sequence. For instance, a person `walking in a circle` is mapped to the two action categories `walk` and `circular movement`. 
    Ideal models will predict high scores for all the categories relevant to the movement  sample. 
3. **Top-1-norm** is the mean `Top-1` across categories. The magnitude of `Top-1-norm` - `Top-1` illustrates the class-specific bias in the model performance. In Babel, it reflects the impact of class imbalance on learning. 


### References 

[1] Shahroudy, Amir, et al. "NTU RGB+D: A large scale dataset for 3d human activity analysis." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. <br>
[2] Shi, Lei, et al. "Two-stream adaptive graph convolutional networks for skeleton-based action recognition." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019. <br>
[3] Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE international conference on computer vision. 2017. <br>
[4] Cui, Yin, et al. "Class-balanced loss based on effective number of samples." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019. <br>
