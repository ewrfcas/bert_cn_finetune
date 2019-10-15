# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import numpy as np
import six
import tensorflow as tf


def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True, *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
       float32 precision and then casts them to the training precision.
    """
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable


def get_custom_getter(compute_type):
    return float32_variable_storage_getter if compute_type == tf.float16 else None


# define the dense layer

try:
    import blocksparse as bs


    def layer_norm(x, name='LayerNorm', epsilon=1e-5, relu=False):
        """
        normalize state vector to be zero mean / unit variance + learned scale/shift
        """
        n_state = x.shape[-1].value
        with tf.variable_scope(name):
            gain = tf.get_variable('gamma', [n_state], initializer=tf.constant_initializer(1.0))
            bias = tf.get_variable('beta', [n_state], initializer=tf.constant_initializer(0.0))

            return bs.layer_norm(x, gain, bias, axis=-1, epsilon=epsilon, relu=relu)


    def dense(x, hidden_size, activation=None, name='dense', kernel_initializer=None, bias=True):
        if kernel_initializer is None:
            kernel_initializer = create_initializer(0.02)
        with tf.variable_scope(name):
            nx = x.shape[-1].value
            ndims = x.shape.ndims
            dtype = x.dtype

            # Note: param initializers are not particularly well tuned in this code
            w = tf.get_variable("kernel", [nx, hidden_size], initializer=kernel_initializer,
                                dtype=dtype)

            assert x.op.device != ''

            if bias:
                b = tf.get_variable("bias", [hidden_size], initializer=tf.zeros_initializer)
            else:
                b = 0

            # merge context and batch dims for more efficient matmul
            if ndims > 2:
                y_shape = tf.concat([tf.shape(x)[: ndims - 1], [hidden_size]], axis=0)
                x = tf.reshape(x, [-1, nx])

            y = tf.matmul(x, w)

            if activation == 'fast_gelu' or activation == 'gelu':
                fast_gelu = True
            else:
                fast_gelu = False
            if activation == 'relu':
                relu = True
            else:
                relu = False
            y = bs.bias_relu(y, b, relu=relu, fast_gelu=fast_gelu, atomics=False)

            if activation == 'tanh':
                y = tf.tanh(y)
            elif activation == 'sigmoid':
                y = tf.sigmoid(y)

            if ndims > 2:
                y = tf.reshape(y, y_shape)

            return y


    def attention_softmax(qk_scores, scale):
        return bs.softmax(qk_scores, scale)

except:
    print('WARNING!!!!Please install blocksparse for faster training and lower gpu memory cost!!!!!!')


    def layer_norm_ops(x, g, b, axis=1, segments=1, epsilon=1e-6):
        if axis < 0:
            axis += len(x.shape)

        K = x.shape[axis].value
        assert g.shape.num_elements() == K
        assert b.shape.num_elements() == K
        assert K % segments == 0
        assert axis != 0 or segments == 1, "Segments only implemented on axis=1 for now"
        K //= segments

        ys = list()
        for s in range(segments):
            segK = slice(s * K, s * K + K)
            segX = [segK if d == axis else slice(None) for d in range(x.shape.ndims)]

            mean, var = tf.nn.moments(x[segX], [axis], keep_dims=True)
            norm = (x[segX] - mean) * tf.rsqrt(var + epsilon)
            ys.append(norm * g[segK] + b[segK])

        y = tf.concat(ys, axis) if segments > 1 else ys[0]

        return y


    def layer_norm(input_tensor, name='LayerNorm', epsilon=1e-5):
        """
        normalize state vector to be zero mean / unit variance + learned scale/shift
        """
        n_state = input_tensor.shape[-1].value
        with tf.variable_scope(name):
            gain = tf.get_variable('gamma', [n_state], initializer=tf.constant_initializer(1.0),
                                   dtype=input_tensor.dtype)
            bias = tf.get_variable('beta', [n_state], initializer=tf.constant_initializer(0.0),
                                   dtype=input_tensor.dtype)
            x = layer_norm_ops(input_tensor, gain, bias, axis=-1, epsilon=epsilon)
            return x


    def dense(x, hidden_size, activation=None, name='dense', kernel_initializer=None, bias=True):
        def gelu(x):
            cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
            return x * cdf

        def fast_gelu(x):
            return x * tf.nn.sigmoid(1.702 * x)

        if kernel_initializer is None:
            kernel_initializer = create_initializer(0.02)
        with tf.variable_scope(name):
            nx = x.shape[-1].value
            ndims = x.shape.ndims
            dtype = x.dtype

            # Note: param initializers are not particularly well tuned in this code
            w = tf.get_variable("kernel", [nx, hidden_size], initializer=kernel_initializer,
                                dtype=dtype)
            if bias:
                b = tf.get_variable("bias", [hidden_size], initializer=tf.zeros_initializer, dtype=dtype)
            else:
                b = 0

            # merge context and batch dims for more efficient matmul
            if ndims > 2:
                y_shape = tf.concat([tf.shape(x)[: ndims - 1], [hidden_size]], axis=0)
                x = tf.reshape(x, [-1, nx])

            y = tf.matmul(x, w)

            if bias:
                y += b

            if activation == 'tanh':
                y = tf.tanh(y)
            elif activation == 'sigmoid':
                y = tf.sigmoid(y)
            elif activation == 'relu':
                y = tf.nn.relu(y)
            elif activation == 'gelu':
                y = gelu(y)
            elif activation == 'fast_gelu':
                y = fast_gelu(y)

            if ndims > 2:
                y = tf.reshape(y, y_shape)

            return y


    def attention_softmax(qk_scores, scale):
        return tf.nn.softmax(qk_scores * scale, axis=-1)


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, rate=dropout_prob)
    return output


def layer_norm_and_dropout(input_tensor, dropout_prob, name='LayerNorm'):
    """Runs layer normalization followed by dropout."""
    output_tensor = layer_norm(input_tensor, name)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def get_shape_list(tensor, expected_rank=None, name=None):
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def split_states(x, n):
    """
    reshape (batch, pixel, state) -> (batch, pixel, head, head_state)
    """
    x_shape = get_shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1] + [n, m // n]
    return tf.reshape(x, new_x_shape)


def split_heads(x, n):
    return tf.transpose(split_states(x, n), [0, 2, 1, 3])


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def merge_states(x):
    """
    reshape (batch, pixel, head, head_state) -> (batch, pixel, state)
    """
    x_shape = get_shape_list(x)
    new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)


def merge_heads(x):
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_float16=False):
    embedding_table = tf.get_variable(
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=create_initializer(initializer_range),
        dtype=tf.float16 if use_float16 else tf.float32)

    output = tf.nn.embedding_lookup(embedding_table, input_ids)
    return output, embedding_table


def embedding_postprocessor(input_tensor,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1,
                            use_float16=False):
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = input_tensor
    token_type_table = tf.get_variable(
        name=token_type_embedding_name,
        shape=[token_type_vocab_size, width],
        initializer=create_initializer(initializer_range),
        dtype=tf.float16 if use_float16 else tf.float32)
    token_type_embeddings = tf.nn.embedding_lookup(token_type_table, token_type_ids)
    output += token_type_embeddings

    full_position_embeddings = tf.get_variable(
        name=position_embedding_name,
        shape=[max_position_embeddings, width],
        initializer=create_initializer(initializer_range),
        dtype=tf.float16 if use_float16 else tf.float32)
    pos_ids = tf.expand_dims(tf.range(seq_length), 0)
    pos_ids = tf.tile(pos_ids, (batch_size, 1))
    position_embeddings = tf.nn.embedding_lookup(full_position_embeddings, pos_ids)
    output += position_embeddings

    output = layer_norm_and_dropout(output, dropout_prob)
    return output


def attention_layer(x, attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02):
    q = dense(x, num_attention_heads * size_per_head, name='query',
              kernel_initializer=create_initializer(initializer_range))
    k = dense(x, num_attention_heads * size_per_head, name='key',
              kernel_initializer=create_initializer(initializer_range))
    v = dense(x, num_attention_heads * size_per_head, name='value',
              kernel_initializer=create_initializer(initializer_range))
    q = split_heads(q, num_attention_heads)
    k = split_heads(k, num_attention_heads)
    v = split_heads(v, num_attention_heads)

    qk = tf.matmul(q, k, transpose_b=True)  # [bs, head, len, len]
    qk += (-10000. * (1 - attention_mask))
    qk = attention_softmax(qk, scale=1.0 / np.sqrt(size_per_head))
    # 本来dropout在这，显存占用大
    qkv = tf.matmul(qk, v)  # [bs, head, len, dim]
    att = merge_heads(qkv)  # [bs, len, dim*head]
    # dropout转移到这里
    att = dropout(att, attention_probs_dropout_prob)

    return att


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn='gelu',
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)

    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            with tf.variable_scope("attention"):
                with tf.variable_scope("self"):
                    attention_head = attention_layer(x=input_tensor,
                                                     attention_mask=attention_mask,
                                                     num_attention_heads=num_attention_heads,
                                                     size_per_head=attention_head_size,
                                                     attention_probs_dropout_prob=attention_probs_dropout_prob,
                                                     initializer_range=initializer_range)

                # Run a linear projection of `hidden_size` then add a residual
                # with `layer_input`.
                with tf.variable_scope("output"):
                    attention_output = dense(
                        attention_head,
                        hidden_size,
                        kernel_initializer=create_initializer(initializer_range))
                    attention_output = dropout(attention_output, hidden_dropout_prob)
                    attention_output = layer_norm(attention_output + input_tensor)

            # The activation is only applied to the "intermediate" hidden layer.
            with tf.variable_scope("intermediate"):
                intermediate_output = dense(
                    attention_output,
                    intermediate_size,
                    activation=intermediate_act_fn,
                    kernel_initializer=create_initializer(initializer_range))

            # Down-project back to `hidden_size` then add the residual.
            with tf.variable_scope("output"):
                layer_output = dense(
                    intermediate_output,
                    hidden_size,
                    kernel_initializer=create_initializer(initializer_range))
                layer_output = dropout(layer_output, hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output)
                input_tensor = layer_output
                all_layer_outputs.append(layer_output)

    if do_return_all_layers:
        return all_layer_outputs
    else:
        return layer_output


class BertConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertModel(object):
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_float16=False,
                 scope="bert"):
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        with tf.variable_scope(scope, default_name="bert", reuse=tf.AUTO_REUSE,
                               custom_getter=get_custom_getter(tf.float16 if use_float16 else tf.float32)):
            with tf.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                self.embedding_output, self.embedding_table = embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    word_embedding_name="word_embeddings",
                    use_float16=use_float16)

                # Add positional embeddings and token type embeddings, then layer
                # normalize and perform dropout.
                self.embedding_output = embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    token_type_ids=token_type_ids,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    position_embedding_name="position_embeddings",
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob,
                    use_float16=use_float16)

            with tf.variable_scope("encoder"):
                attention_mask = tf.reshape(input_mask, (-1, 1, 1, input_mask.shape[1]))  # [bs, len]->[bs, 1, 1, len]
                attention_mask = tf.cast(attention_mask, self.embedding_output.dtype)

                # Run the stacked transformer.
                # `sequence_output` shape = [batch_size, seq_length, hidden_size].
                self.all_encoder_layers = transformer_model(
                    input_tensor=self.embedding_output,
                    attention_mask=attention_mask,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    intermediate_act_fn=config.hidden_act,
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    do_return_all_layers=True)

            self.sequence_output = self.all_encoder_layers[-1]
            with tf.variable_scope("pooler"):
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token. We assume that this has been pre-trained
                first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
                self.pooled_output = dense(
                    first_token_tensor,
                    config.hidden_size,
                    activation='tanh',
                    kernel_initializer=create_initializer(config.initializer_range))

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_embedding_output(self):
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table


class BertModelMRC(object):
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 start_positions=None,
                 end_positions=None,
                 use_float16=False,
                 scope="bert"):
        with tf.device("/gpu:0"):
            self.bert = BertModel(config, is_training, input_ids, input_mask, token_type_ids, use_float16, scope)

            # finetune mrc
            with tf.variable_scope('finetune_mrc', reuse=tf.AUTO_REUSE,
                                   custom_getter=get_custom_getter(tf.float16 if use_float16 else tf.float32)):
                self.sequence_output = self.bert.get_sequence_output()
                # [bs, len]
                self.start_logits = tf.squeeze(dense(self.sequence_output, 1, name='start_dense'), -1)
                self.end_logits = tf.squeeze(dense(self.sequence_output, 1, name='end_dense'), -1)
                self.start_logits += tf.cast(-10000. * (1 - input_mask), self.start_logits.dtype)
                self.end_logits += tf.cast(-10000. * (1 - input_mask), self.end_logits.dtype)

                if is_training and start_positions is not None and end_positions is not None:
                    start_loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=tf.cast(self.start_logits, tf.float32),
                        labels=start_positions)
                    end_loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=tf.cast(self.end_logits, tf.float32),
                        labels=end_positions)
                    start_loss = tf.reduce_mean(start_loss_)
                    end_loss = tf.reduce_mean(end_loss_)
                    self.train_loss = (start_loss + end_loss) / 2.0
