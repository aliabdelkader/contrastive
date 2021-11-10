#!/bin/bash


docker run -it --rm -v /home/mrafaat/SemanticKitti:/home/user/SemanticKitti -v /home/mrafaat/AliThesis/logs:/home/user/logs -v `pwd`/../:/home/user/contrastive --network host  --ipc=host --runtime nvidia --gpus '"device=0,1"' contrast_image

