
# BABEL: Bodies, Action and Behavior with English Labels [[CVPR 2021](http://cvpr2021.thecvf.com/)]

<p float="center">
  <img src="https://babel.is.tue.mpg.de/media/upload/babel_teaser.png" width="90%" />
</p>
        
> [Abhinanda R. Punnakkal\*](https://ps.is.tuebingen.mpg.de/person/apunnakkal), [Arjun Chandrasekaran\*](https://ps.is.tuebingen.mpg.de/person/achandrasekaran), [Nikos Athanasiou](https://ps.is.tuebingen.mpg.de/person/nathanasiou), [Alejandra Quiros-Ramirez](https://ps.is.tuebingen.mpg.de/person/aquiros), [Michael J. Black](https://ps.is.tuebingen.mpg.de/person/black). 
> \* denotes equal contribution

[Project Website](https://babel.is.tue.mpg.de) | [Paper](https://arxiv.org/pdf/2106.09696.pdf) | [Video](https://keeper.mpdl.mpg.de/f/9d733e2eb1ec4914970b/) | [Poster](https://babel.is.tue.mpg.de/media/upload/CVPR_2021_BABEL_poster.pdf) 

---

BABEL is a large dataset with language labels describing the actions being performed in mocap sequences. BABEL labels about 43 hours of mocap sequences from [AMASS](https://amass.is.tue.mpg.de/) [1] with action labels. 
Sequences have action labels at two possible levels of abstraction: 
- **Sequence labels** which describe the overall action in the sequence
- **Frame labels** which describe all actions in every frame of the sequence. Each frame label is precisely aligned with the duration of the corresponding action in the mocap sequence, and multiple actions can overlap.

To download the BABEL action labels, visit our ['Data' page](https://babel.is.tue.mpg.de/data.html). You can download the mocap sequences from [AMASS](https://amass.is.tue.mpg.de/). 


### Tutorials 

We release some helper code in Jupyter notebooks to load the BABEL dataset, visualize mocap sequences and their action labels, search BABEL for sequences containing specific actions, etc. 

See [`notebooks/`](notebooks/) for more details. 


### Action Recognition 

We provide features, training and inference code, and pre-trained checkpoints for 3D skeleton-based action recognition. 

Please see [`action_recognition/`](action_recognition/) for more details. 


### Acknowledgements

The notebooks in this repo are inspired by the those provided by [AMASS](https://github.com/nghorbani/amass). 
The Action Recognition code is based on the [2s-AGCN](https://github.com/lshiwjx/2s-AGCN) implementation. 


### References 

[1] Mahmood, Naureen, et al. "AMASS: Archive of motion capture as surface shapes." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.

### License

Software Copyright License for non-commercial scientific research purposes. Please read carefully the terms and conditions and any accompanying documentation before you download and/or use the AMASS dataset, and software, (the "Model & Software"). By downloading and/or using the Model & Software (including downloading, cloning, installing, and any other use of this GitHub repository), you acknowledge that you have read these terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the Model & Software. Any infringement of the terms of this agreement will automatically terminate your rights under this License.

### Contact

The code in this repository is developed by [Abhinanda Punnakkal](https://www.is.mpg.de/person/apunnakkal) and [Arjun Chandrasekaran](https://www.is.mpg.de/person/achandrasekaran).

If you have any questions you can contact us at babel@tue.mpg.de.

