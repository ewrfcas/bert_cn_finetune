import tensorflow as tf
from models.tf_modeling import BertModelMRC, BertConfig
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

bert_config = BertConfig.from_json_file('check_points/pretrain_models/roberta_wwm_ext_large/bert_config.json')
max_seq_length = 512
input_ids = tf.placeholder(tf.int32, shape=[None, max_seq_length], name='input_ids')
segment_ids = tf.placeholder(tf.int32, shape=[None, max_seq_length], name='segment_ids')
input_mask = tf.placeholder(tf.float32, shape=[None, max_seq_length], name='input_masks')
eval_model = BertModelMRC(config=bert_config,
                          is_training=False,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          token_type_ids=segment_ids,
                          use_float16=False)

# load pytorch model
pytorch_weights = torch.load('pytorch_model.pth')
for k in pytorch_weights:
    print(k, pytorch_weights[k].shape)

# print tf parameters
for p in tf.trainable_variables():
    print(p)

convert_ops = []
for p in tf.trainable_variables():
    tf_name = p.name
    if 'kernel' in p.name:
        do_transpose = True
    else:
        do_transpose = False
    pytorch_name = tf_name.strip(':0').replace('layer_','layer.').replace('/','.').replace('gamma','weight')\
    .replace('beta','bias').replace('kernel','weight').replace('_embeddings','_embeddings.weight').replace('output_bias', 'bias')
    if pytorch_name in pytorch_weights:
        print('Convert Success:', tf_name)
        weight = tf.constant(pytorch_weights[pytorch_name].cpu().numpy())
        if weight.dtype == tf.float16:
            weight = tf.cast(weight, tf.float32)
        if do_transpose is True:
            weight = tf.transpose(weight)
        convert_op = tf.assign(p, weight)
        convert_ops.append(convert_op)
    else:
        print('Convert Failed:', tf_name, pytorch_name)

saver = tf.train.Saver(var_list=tf.trainable_variables())
from tqdm import tqdm_notebook as tqdm
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for op in tqdm(convert_ops):
        sess.run(op)
    saver.save(sess, save_path='model.ckpt', write_meta_graph=False)
