# GPT-Talker

## Introduction
This is an implementation of the following paper.
[《Generative Expressive Conversational Speech Synthesis》](https://arxiv.org/pdf/2407.21491)
 (Accepted by MM'2024)

[Rui Liu *](https://ttslr.github.io/), Yifan Hu, [Yi Ren](https://rayeren.github.io/), Xiang Yin, [Haizhou Li](https://colips.org/~eleliha/).

## Demo Page
[Speech Demo](https://walker-hyf.github.io/GPT-Talker/)

## Dependencies
* For details about the operating environment dependency. Please refer to [GPT-SoVITS'requirements.txt](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/requirements.txt)
* Please ```conda install ffmpeg```
* Tested environment: Ubuntu=22.04.2, python=3.9.18, torch=2.0.1+cu118

## NCSSD
The large-scale conversational speech synthesis dataset we constructed, including those collected over the Internet as well as those recorded by sound recorders, consists of approximately 236 hours and over 776 speakers.

Please refer to [NCSSD'repo](https://github.com/walker-hyf/NCSSD)

## Prepare Datasets
Execute the five steps in the [./prepare_datastes](./prepare_datasets/) directory to build the training data for GPT-Talker.

## Train
* Conversational VITS

    ```python train_s2.py```

    The corresponding configuration file is in ./configs/s2.json

* Conversational GPT

    ```python train_s1.py```

    The corresponding configuration file is in ./configs/s1longer.yaml

## Fine-tuning
Fine-tunable base models in the [./pretrained_models](./pretrained_models/), from [GPT-SoVITS](https://drive.google.com/drive/folders/15rap3Z_-w0mYgxz66pDcx2abhDRb17dk?usp=sharing) (Single Speech).

## Citations

```bibtex
@inproceedings{
liu2024generative,
    title={Generative Expressive Conversational Speech Synthesis},
    author={Rui Liu and Yifan Hu and Yi Ren and Xiang Yin and Haizhou Li},
    booktitle={ACM Multimedia 2024},
    year={2024},
    url={https://openreview.net/forum?id=eK9ShhDqwu}
}
```
