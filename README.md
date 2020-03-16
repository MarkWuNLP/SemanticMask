# ASR_SemanticMask
The repo contains our code of ``Semantic Mask for Transformer based End-to-End Speech Recognition"

## Preparation
We already build a runnable docker, you can run the following command to download and run the docker

`docker run -it --volume-driver=nfs --shm-size=64G j4ckl1u/espnet-py36-img:latest /bin/bash`

Regarding data preparation, I suggest you read [ESPnet instructions](https://github.com/espnet/espnet/tree/master/egs/librispeech). It should be note that espnet doesn't do speed perturbation, but I strongly recommend to do it according to the better performance on dev-other and test-other datasets. 

### Word Alignment
To enable semantic mask training, you have to align audio and word with a word alignment tool. 
(Todo Chengyi add details for word alignment)

## Training and decoding
For training, I upload my training configs into configs folder, including base setting and large setting respectively. Our archtecture is similar to ESPnet, but replacing position embedding with CNN in both encoder and decoder. The specific code change can be found at [here](https://github.com/MarkWuNLP/SemanticMask/blob/master/espnet/nets/pytorch_backend/transformer/subsampling.py)

In terms of decoding, pleaes first download the ESPnet [pre-trained RNN language model](https://github.com/espnet/espnet/tree/master/egs/librispeech), and then run our decoding script to get the model output. 
## Pre-train Models
We release a [base model](https://drive.google.com/drive/folders/1qQKVx3jBxIB_zII7wym9o9RK7G9Gb5aY?usp=sharing) (12 encoder layers and 6 decoder layers) and a [large model](https://drive.google.com/drive/folders/12lcFfpvD-sJpqi0T2xfudoZT9QjLaRei?usp=sharing) (24 encoder layers and 12 decoder layers). It achevies following results with shallow language model fusion setting.


|      |dev-clean|dev-other|test-clean|test-other| 
| ------------- |:-------------:|:-------------:|:-------------:|:-------------:|
| Base  | 2.07 | 5.06| 2.31|5.21 |
| Large     | 2.02|4.91| 2.19  |5.19 |
