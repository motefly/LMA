#  Light Multi-segment Activation for Model Compression

Implementation for the paper "Light Multi-segment Activation for model compression", which has been submitted to NeurIPS'2019.

**Note: We are sorry that there are two minor issues in the Supplementary File submitted, 1) In the legend of the last figure, that "LMA-S2" and "ReLU-S2" are reversed, 2) In the Table 6, that the given parameter numbers are wrong. The corrected ones are shown on the bottom of this page.**

# Intorduction

This repo is built for the experimental codes in our paper, containing all the model implementation, data acquisition, and parameter settings. Here we thanks the authors of the [codebase](https://github.com/antspy/quantized_distillation), and our repo is improved from it.

There are three main functional folders, `cnn_models`, `translation_models` and `onmt`. `cnn_models` contains all the detailed implementations of CNN architectures; `translation_models` contains the high-level definition of translational models and `onmt` is part of the [OpenNMT-py codebase](https://github.com/OpenNMT/OpenNMT-py), containing much detailed translation-related implementation (transformers etc.). Besides, after running our codes, some folders named `summary`, `models`, `datas`, `manager` for some result storages may be created.

Moreover, there are three `main_*.py` for the experimental entries. `main.py` is for the experiments on CIFAR-10, `main_wrn.py` is for running Wide Residual Networks on CIFAR-100, `main_opennmt` is for the translational experiments on both Ope and WMT13, `main_joint.py` is used for testing the effectiveness when jointly using Quantized Distillation and LMA. Besides, `activations.py` contains the implementations of LMA and 
the other baseline activations. `model_manager.py` mainly supports the distillation framework, which can well manage the teacher model and student models.

FYI, `scripts` is for batch running, `perl_scripts` is for BLEU computation; `helpers` contains some common utils; `datasets` is for data acquisition, which supports download and decompresses the datas automatically; `quantization` is mainly succeeded from the original codebase and can be used for joint experiments.

# Getting started

## Environment setup

The models are based on Python and Pytorch, to run the codes, please set up the environment first:
1. Install Anaconda with Python >= 3.6. [Miniconda](https://conda.io/miniconda.html) is a quick way to get started.
2. Clone the repository
    ```
    git clone https://github.com/LMA-NeurIPS19/LMA
    ```
3. Run the conda file script to create a conda environment:
    ```
    cd LMA
    conda env create -f environment.yml  
    ```

## Training teachers

To train a teacher model, using the following script:
```bash
python main.py -train_teacher -manager 0523
```
It will produce a new trained teacher model saved in `models` and the model manager will be saved in `manager`. The arguments not specified are set to the default settings in the main file. And more default settings for translational models are [here](onmt/standard_options.py).


## Training students

After the above step, we can train a student model under the supervision of from the teacher and ground truth. For example,
```bash
for sed in 1 2 3 4 5
do
for stmodel in 0 1 2
do
    python main.py -train_student -manager 0523-stModel $stmodel -stud_act relu -plot_title 0523 -seed $sed
    python main.py -train_student -manager 0523 -stModel $stmodel -stud_act lma -plot_title 0523 -seed $sed
    python main.py -train_student -manager 0523 -stModel $stmodel -stud_act swish -plot_title 0523 -seed $sed
    python main.py -train_student -manager 0523 -stModel $stmodel -stud_act aplu -plot_title 0523 -seed $sed -num_bins 8
    python main.py -train_student -manager 0523 -stModel $stmodel -stud_act prelu -plot_title 0523 -seed $sed -num_bins 8
done
done

```
## Example Results

### Numerical results with different random seeds

The following table shows the example results by running the scripts on CIFAR-10 with Student 1.

| Method | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 | Mean | Std |
|--|--|--|--|--|--|--|--|
| ReLU | 88.64 |88.91 | 88.35 | 88.98 | 88.83 | 89.92 | 0.21 |
| PReLU | 89.49 | 89.20 | 89.41 | 89.69 | 88.76 | 89.31 | 0.35 | 
| Swish | 89.18 | 88.92 | 88.97 | 89.11 | 89.00 | 89.03 | 0.11 |
| APLU-8 | 89.67 | 90.12 | 89.75 | 89.94 | 90.13 | 89.92 | 0.21 |
| LMA-8 | 90.59 | 90.27 | 90.84 | 90.62 | 90.53 | 90.57 | 0.20 |

### Convergence curves on CIFAR-100

Here we show the corrected figure that will replace the one in the Supplementary File.

![Testing Accuracy-Epoch Curves on CIFAR-100](images/cifar100-acc-fix.png)

The correct Table 6 in the Supplementary File should be,

| Model | Widen Factor | Depth | Parameter Number |
|--|--|--|--|
| Teacher Model | 10 | 16 | 17.2 M |
| Student Model 1 | 6 | 10 | 1.22 M |
| Student Model 2 | 4 | 10 | 0.32 M |

Sorry again and thanks for your visiting, if you have any questions, please new an issue.
