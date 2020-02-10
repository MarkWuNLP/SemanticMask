# ASR_SemanticMask
The repo contains our code of ``Semantic Mask for Transformer based End-to-End Speech Recognition"

## Preparation
We already build a runnable docker, you can run the following commond to download and run the docker

`docker run -it --volume-driver=nfs --shm-size=64G j4ckl1u/espnet-py36-img:latest /bin/bash`

Regarding to data preparation, I suggest you read [ESPnet instructions](https://github.com/espnet/espnet/tree/master/egs/librispeech). It should be note that espnet doesn't do speed perturbation, but I strongly recommend to do it according to the better performance on dev-other and test-other datasets. 

## Training and decoding 
## Pre-train Models
We release a [base model](https://drive.google.com/open?id=1tQVX24aN5NpOtDFWO6ZsVWOLHuOjEj8W) (12 encoder layers and 6 decoder layers) and a [large model](https://drive.google.com/open?id=1zDS_cUhyo17foGMsbUBERuI8u1jC13vB) (24 encoder layers and 12 decoder layers).
