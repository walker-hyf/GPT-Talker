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
@inproceedings{10.1145/3664647.3681697,
  author = {Liu, Rui and Hu, Yifan and Ren, Yi and Yin, Xiang and Li, Haizhou},
  title = {Generative Expressive Conversational Speech Synthesis},
  year = {2024},
  isbn = {9798400706868},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3664647.3681697},
  doi = {10.1145/3664647.3681697},
  abstract = {Conversational Speech Synthesis (CSS) aims to express a target utterance with the proper speaking style in a user-agent conversation setting. Existing CSS methods employ effective multi-modal context modeling techniques to achieve empathy understanding and expression. However, they often need to design complex network architectures and meticulously optimize the modules within them. In addition, due to the limitations of small-scale datasets containing scripted recording styles, they often fail to simulate real natural conversational styles. To address the above issues, we propose a novel generative expressive CSS system, termed GPT-Talker.We transform the multimodal information of the multi-turn dialogue history into discrete token sequences and seamlessly integrate them to form a comprehensive user-agent dialogue context. Leveraging the power of GPT, we predict the token sequence, that includes both semantic and style knowledge, of response for the agent. After that, the expressive conversational speech is synthesized by the conversation-enriched VITS to deliver feedback to the user.Furthermore, we propose a large-scale Natural CSS Dataset called NCSSD, that includes both naturally recorded conversational speech in improvised styles and dialogues extracted from TV shows. It encompasses both Chinese and English languages, with a total duration of 236 hours. We conducted comprehensive experiments on the reliability of the NCSSD and the effectiveness of our GPT-Talker. Both subjective and objective evaluations demonstrate that our model outperforms other state-of-the-art CSS systems significantly in terms of naturalness and expressiveness. The Code, Dataset, and Pre-trained Model are available at: https://github.com/AI-S2-Lab/GPT-Talker.},
  booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
  pages = {4187–4196},
  numpages = {10},
  keywords = {conversational speech synthesis (css), expressiveness, gpt, user-agent conversation},
  location = {Melbourne VIC, Australia},
  series = {MM '24}
}
```
