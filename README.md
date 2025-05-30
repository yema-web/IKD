# IKD

This repo:

**(1) covers the implementation of the following AAAI 2025 paper:**

"Debiased Distillation for Consistency Regularization" (IKD). [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/32840).

<div style="text-align:center"><img src="https://github.com/yema-web/IKD/tree/main"

<p></p>

## Installation

This repo was tested with Ubuntu 16.04.5 LTS, Python 3.5, PyTorch 0.4.0, and CUDA 9.0. But it should be runnable with recent PyTorch versions >=0.4.0

## Running

1. Fetch the pretrained teacher models by:

    ```
    sh scripts/fetch_pretrained_teachers.sh
    ```
   which will download and save the models to `save/models`
   
2. Run distillation by following commands in `scripts/run_cifar_distill.sh`. An example of running Geoffrey's original Knowledge Distillation (KD) is given by:

    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --trial 1
    ```
    where the flags are explained as:
    - `--path_t`: specify the path of the teacher model
    - `--model_s`: specify the student model, see 'models/\_\_init\_\_.py' to check the available model types.
    - `--distill`: specify the distillation method
    - `-r`: the weight of the cross-entropy loss between logit and ground truth, default: `1`
    - `-a`: the weight of the KD loss, default: `None`
    - `-b`: the weight of other distillation losses, default: `None`
    - `--trial`: specify the experimental id to differentiate between multiple runs.
    
    Therefore, the command for running IKD is something like:
    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8x4 -a 0 -b 0.8 --trial 1
    ```
    
3. Combining a distillation objective with DKD is simply done by setting `-a` as a non-zero value, which results in the following example (combining DKD with IKD)
    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill dkd --model_s resnet8x4 -a 1.0 -b 8.0 --avg 1.0 --trial 1     
    ```

4. (optional) Train teacher networks from scratch. Example commands are in `scripts/run_cifar_vanilla.sh`

Note: the default setting is for a single-GPU training. If you would like to play this repo with multiple GPUs, you might need to tune the learning rate, which empirically needs to be scaled up linearly with the batch size, see [this paper](https://arxiv.org/abs/1706.02677)



## Citation

If you find this repo useful for your research, please consider citing the paper

```
@inproceedings{tian2019crd,
  title={Contrastive Representation Distillation},
  author={Yonglong Tian and Dilip Krishnan and Phillip Isola},
  booktitle={International Conference on Learning Representations},
  year={2020}
}

@inproceedings{wang2025debiased,
  title={Debiased Distillation for Consistency Regularization},
  author={Wang, Lu and Xu, Liuchi and Yang, Xiong and Huang, Zhenhua and Cheng, Jun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={8},
  pages={7799--7807},
  year={2025}
}
```

## Acknowledgement