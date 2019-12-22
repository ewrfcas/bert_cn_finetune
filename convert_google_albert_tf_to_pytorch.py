# coding=utf-8
# Copyright 2018 The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert BERT checkpoint."""

from __future__ import print_function

import os
import re
import argparse
import tensorflow as tf
import torch
import numpy as np
import ipdb

from models.google_albert_pytorch_modeling import AlbertConfig, AlbertForPreTraining, AlbertForMRC


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    print("Converting TensorFlow checkpoint from {} with config at {}".format(tf_checkpoint_path, bert_config_file))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_checkpoint_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_checkpoint_path, name)
        names.append(name)
        arrays.append(array)

    # Initialise PyTorch model
    config = AlbertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = AlbertForMRC(config)

    for name, array in zip(names, arrays):
        name = name.replace('group_0/inner_group_0/', '')
        name = name.split('/')
        if name[0] == 'global_step' or name[0] == 'cls':  # or name[0] == 'finetune_mrc'
            continue
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            # if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
            #     l = re.split(r'_(\d+)', m_name)
            # else:
            #     l = [m_name]
            l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            # if len(l) >= 2:
            #     num = int(l[1])
            #     pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
            # array = np.transpose(array)
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            print(name, 'SHAPE WRONG!')
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--tf_checkpoint_path",
                        default='check_points/pretrain_models/albert_xxlarge_google_zh_v1121/model.ckpt-best',
                        type=str,
                        help="Path the TensorFlow checkpoint path.")
    parser.add_argument("--bert_config_file",
                        default='check_points/pretrain_models/albert_xxlarge_google_zh_v1121/bert_config.json',
                        type=str,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--pytorch_dump_path",
                        default='check_points/pretrain_models/albert_xxlarge_google_zh_v1121/pytorch_model.pth',
                        type=str,
                        help="Path to the output PyTorch model.")
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path,
                                     args.bert_config_file,
                                     args.pytorch_dump_path)
