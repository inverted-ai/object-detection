# Object detection
Provided files here are additional files required for training [PGD](https://arxiv.org/abs/2107.14160) model on `baseline` and `INITIALIZE` datasets.
These files need to be added to [branch 1.0](https://github.com/open-mmlab/mmdetection3d/tree/1.0) from mmdetection3d repository.
- Install [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/blob/1.0/docs/en/getting_started.md)
- Add `carla_utils` folder to `mmdetection3d/mmdet3d/core/evaluation/` directory.
- Add `carla_mono_dataset.py` to `mmdetection3d/mmdet3d/datasets` directory.
- Add `configs/datasets/` to `mmdetection3d/configs/_base_/datasets/` directory.
- Add `configs/pgd` to `mmdetection3d/configs/pgd/` directory.
