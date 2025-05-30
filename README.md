##Debiased Distillation for Consistency Regularization

This repo:

"Debiased Distillation for Consistency Regularization" ( accepted by AAAI 2025 paper). [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/32840).

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
    
      
3. Combining DKD with IKD
    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill dkd --model_s resnet8x4 -r 1.0 -a 1.0 -b 8.0 --avg 6.5 --intra_T 1.0 --trial 1     
    ```
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