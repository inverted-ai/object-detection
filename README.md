# object-detection
This repository provides the code and datasets introduced in ["Realistically distributing object placements in synthetic training data improves the performance of vision-based object detection models"](https://arxiv.org/abs/2305.14621). 

Datasets generated using the [CARLA driving simulator](https://github.com/carla-simulator/carla). In generating `baseline` dataset  we allow the CARLA Traffic Manager to freely move vehicles and take a snapshot of their positions at a particular time. While `INITIALIZE` dataset utilizes a [state-of-the-art commercial model](https://docs.inverted.ai/) that jointly samples realistic vehicle placements.

We use the [PGD model](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/pgd) to train two object detection models on `baseline` and `INITIALIZE` datasets. The code for data loading and training configurations are provided. 

```
@misc{dabiri2023realistically,
      title={Realistically distributing object placements in synthetic training data improves the performance of vision-based object detection models}, 
      author={Setareh Dabiri and Vasileios Lioutas and Berend Zwartsenberg and Yunpeng Liu and Matthew Niedoba and Xiaoxuan Liang and Dylan Green and Justice Sefas and Jonathan Wilder Lavington and Frank Wood and Adam Scibior},
      year={2023},
      eprint={2305.14621},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
