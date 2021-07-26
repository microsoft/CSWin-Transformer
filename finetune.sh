# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Xiaoyi Dong
# ------------------------------------------

NUM_PROC=$1
shift
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC finetune.py "$@"

