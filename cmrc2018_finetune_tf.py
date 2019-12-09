import argparse
import os

import numpy as np
import tensorflow as tf

try:
    # horovod must be import before optimizer!
    import horovod.tensorflow as hvd
except:
    print('Please setup horovod before using multi-gpu!!!')
    hvd = None

from models.tf_modeling import BertModelMRC, BertConfig
from optimizations.tf_optimization import Optimizer
import json
import utils
from evaluate.cmrc2018_evaluate import get_eval
from evaluate.cmrc2018_output import write_predictions
import random
from tqdm import tqdm
import collections
from tokenizations.official_tokenization import BertTokenizer
from preprocess.cmrc2018_preprocess import json2features


def print_rank0(*args):
    if mpi_rank == 0:
        print(*args, flush=True)


def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        session = session._sess
    return session


def data_generator(data, n_batch, shuffle=False, drop_last=False):
    steps_per_epoch = len(data) // n_batch
    if len(data) % n_batch != 0 and not drop_last:
        steps_per_epoch += 1
    data_set = dict()
    for k in data[0]:
        data_set[k] = np.array([data_[k] for data_ in data])
    index_all = np.arange(len(data))

    while True:
        if shuffle:
            random.shuffle(index_all)
        for i in range(steps_per_epoch):
            yield {k: data_set[k][index_all[i * n_batch:(i + 1) * n_batch]] for k in data_set}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser.add_argument('--gpu_ids', type=str, default='2')

    # training parameter
    parser.add_argument('--train_epochs', type=int, default=2)
    parser.add_argument('--n_batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--loss_scale', type=float, default=2.0 ** 15)
    parser.add_argument('--warmup_rate', type=float, default=0.1)
    parser.add_argument('--loss_count', type=int, default=1000)
    parser.add_argument('--seed', type=list, default=[123, 456, 789, 556, 977])
    parser.add_argument('--float16', type=int, default=True)  # only sm >= 7.0 (tensorcores)
    parser.add_argument('--max_ans_length', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=30)  # show the average loss per 30 steps args.
    parser.add_argument('--n_best', type=int, default=20)
    parser.add_argument('--eval_epochs', type=float, default=0.5)
    parser.add_argument('--save_best', type=bool, default=True)
    parser.add_argument('--vocab_size', type=int, default=21128)
    parser.add_argument('--max_seq_length', type=int, default=512)

    # data dir
    parser.add_argument('--vocab_file', type=str,
                        default='check_points/pretrain_models/google_bert_base/vocab.txt')

    parser.add_argument('--train_dir', type=str, default='dataset/cmrc2018/train_features_roberta512.json')
    parser.add_argument('--dev_dir1', type=str, default='dataset/cmrc2018/dev_examples_roberta512.json')
    parser.add_argument('--dev_dir2', type=str, default='dataset/cmrc2018/dev_features_roberta512.json')
    parser.add_argument('--train_file', type=str, default='origin_data/cmrc2018/cmrc2018_train.json')
    parser.add_argument('--dev_file', type=str, default='origin_data/cmrc2018/cmrc2018_dev.json')
    parser.add_argument('--bert_config_file', type=str,
                        default='check_points/pretrain_models/google_bert_base/bert_config.json')
    parser.add_argument('--init_restore_dir', type=str,
                        default='check_points/pretrain_models/google_bert_base/bert_model.ckpt')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='check_points/cmrc2018/google_bert_base/')
    parser.add_argument('--setting_file', type=str, default='setting.txt')
    parser.add_argument('--log_file', type=str, default='log.txt')

    # use some global vars for convenience
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    n_gpu = len(args.gpu_ids.split(','))
    if n_gpu > 1:
        assert hvd
        hvd.init()
        mpi_size = hvd.size()
        mpi_rank = hvd.local_rank()
        assert mpi_size == n_gpu
        training_hooks = [hvd.BroadcastGlobalVariablesHook(0)]
        print_rank0('GPU NUM', n_gpu)
    else:
        hvd = None
        mpi_size = 1
        mpi_rank = 0
        training_hooks = None
        print('GPU NUM', n_gpu)

    args.checkpoint_dir += ('/epoch{}_batch{}_lr{}_warmup{}_anslen{}_tf/'
                            .format(args.train_epochs, args.n_batch, args.lr, args.warmup_rate, args.max_ans_length))
    args = utils.check_args(args, mpi_rank)
    print_rank0('######## generating data ########')

    if mpi_rank == 0:
        tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
        # assert args.vocab_size == len(tokenizer.vocab)
        if not os.path.exists(args.train_dir):
            json2features(args.train_file, [args.train_dir.replace('_features_', '_examples_'),
                                            args.train_dir], tokenizer, is_training=True)

        if not os.path.exists(args.dev_dir1) or not os.path.exists(args.dev_dir2):
            json2features(args.dev_file, [args.dev_dir1, args.dev_dir2], tokenizer, is_training=False)

    train_data = json.load(open(args.train_dir, 'r'))
    dev_examples = json.load(open(args.dev_dir1, 'r'))
    dev_data = json.load(open(args.dev_dir2, 'r'))

    if mpi_rank == 0:
        if os.path.exists(args.log_file):
            os.remove(args.log_file)

    # split_data for multi_gpu
    if n_gpu > 1:
        np.random.seed(np.sum(args.seed))
        np.random.shuffle(train_data)
        data_split_start = int(len(train_data) * (mpi_rank / mpi_size))
        data_split_end = int(len(train_data) * ((mpi_rank + 1) / mpi_size))
        train_data = train_data[data_split_start:data_split_end]
        args.n_batch = args.n_batch // n_gpu
        print('#### Hvd rank', mpi_rank, 'train from', data_split_start,
              'to', data_split_end, 'Data length', len(train_data))

    steps_per_epoch = len(train_data) // args.n_batch
    eval_steps = int(steps_per_epoch * args.eval_epochs)
    dev_steps_per_epoch = len(dev_data) // (args.n_batch * n_gpu)
    if len(train_data) % args.n_batch != 0:
        steps_per_epoch += 1
    if len(dev_data) % (args.n_batch * n_gpu) != 0:
        dev_steps_per_epoch += 1
    total_steps = steps_per_epoch * args.train_epochs
    warmup_iters = int(args.warmup_rate * total_steps)

    print_rank0('steps per epoch:', steps_per_epoch)
    print_rank0('total steps:', total_steps)
    print_rank0('warmup steps:', warmup_iters)

    F1s = []
    EMs = []
    best_f1_em = 0
    with tf.device("/gpu:0"):
        input_ids = tf.placeholder(tf.int32, shape=[None, args.max_seq_length], name='input_ids')
        input_masks = tf.placeholder(tf.float32, shape=[None, args.max_seq_length], name='input_masks')
        segment_ids = tf.placeholder(tf.int32, shape=[None, args.max_seq_length], name='segment_ids')
        start_positions = tf.placeholder(tf.int32, shape=[None, ], name='start_positions')
        end_positions = tf.placeholder(tf.int32, shape=[None, ], name='end_positions')

    # build the models for training and testing/validation
    print_rank0('######## init model ########')
    bert_config = BertConfig.from_json_file(args.bert_config_file)
    train_model = BertModelMRC(config=bert_config,
                               is_training=True,
                               input_ids=input_ids,
                               input_mask=input_masks,
                               token_type_ids=segment_ids,
                               start_positions=start_positions,
                               end_positions=end_positions,
                               use_float16=args.float16)

    eval_model = BertModelMRC(config=bert_config,
                              is_training=False,
                              input_ids=input_ids,
                              input_mask=input_masks,
                              token_type_ids=segment_ids,
                              use_float16=args.float16)

    optimization = Optimizer(loss=train_model.train_loss,
                             init_lr=args.lr,
                             num_train_steps=total_steps,
                             num_warmup_steps=warmup_iters,
                             hvd=hvd,
                             use_fp16=args.float16,
                             loss_count=args.loss_count,
                             clip_norm=args.clip_norm,
                             init_loss_scale=args.loss_scale)

    if mpi_rank == 0:
        saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=1)
    else:
        saver = None

    for seed_ in args.seed:
        best_f1, best_em = 0, 0
        if mpi_rank == 0:
            with open(args.log_file, 'a') as aw:
                aw.write('===================================' +
                         'SEED:' + str(seed_)
                         + '===================================' + '\n')
        print_rank0('SEED:', seed_)
        # random seed
        np.random.seed(seed_)
        random.seed(seed_)
        tf.set_random_seed(seed_)

        train_gen = data_generator(train_data, args.n_batch, shuffle=True, drop_last=False)
        dev_gen = data_generator(dev_data, args.n_batch * n_gpu, shuffle=False, drop_last=False)

        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = str(mpi_rank)
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        utils.show_all_variables(rank=mpi_rank)
        utils.init_from_checkpoint(args.init_restore_dir, rank=mpi_rank)
        RawResult = collections.namedtuple("RawResult",
                                           ["unique_id", "start_logits", "end_logits"])

        with tf.train.MonitoredTrainingSession(checkpoint_dir=None,
                                               hooks=training_hooks,
                                               config=config) as sess:
            old_global_steps = sess.run(optimization.global_step)
            for i in range(args.train_epochs):
                print_rank0('Starting epoch %d' % (i + 1))
                total_loss = 0
                iteration = 0
                with tqdm(total=steps_per_epoch, desc='Epoch %d' % (i + 1),
                          disable=False if mpi_rank == 0 else True) as pbar:
                    while iteration < steps_per_epoch:
                        batch_data = next(train_gen)
                        feed_data = {input_ids: batch_data['input_ids'],
                                     input_masks: batch_data['input_mask'],
                                     segment_ids: batch_data['segment_ids'],
                                     start_positions: batch_data['start_position'],
                                     end_positions: batch_data['end_position']}
                        loss, _, global_steps, loss_scale = sess.run(
                            [train_model.train_loss, optimization.train_op, optimization.global_step,
                             optimization.loss_scale],
                            feed_dict=feed_data)
                        if global_steps > old_global_steps:
                            old_global_steps = global_steps
                            total_loss += loss
                            pbar.set_postfix({'loss': '{0:1.5f}'.format(total_loss / (iteration + 1e-5))})
                            pbar.update(1)
                            iteration += 1
                        else:
                            print_rank0('NAN loss in', iteration, ', Loss scale reduce to', loss_scale)

                        if global_steps % eval_steps == 0 and global_steps > 1:
                            print_rank0('Evaluating...')
                            all_results = []
                            for i_step in tqdm(range(dev_steps_per_epoch),
                                               disable=False if mpi_rank == 0 else True):
                                batch_data = next(dev_gen)
                                feed_data = {input_ids: batch_data['input_ids'],
                                             input_masks: batch_data['input_mask'],
                                             segment_ids: batch_data['segment_ids']}
                                batch_start_logits, batch_end_logits = sess.run(
                                    [eval_model.start_logits, eval_model.end_logits],
                                    feed_dict=feed_data)
                                for j in range(len(batch_data['unique_id'])):
                                    start_logits = batch_start_logits[j]
                                    end_logits = batch_end_logits[j]
                                    unique_id = batch_data['unique_id'][j]
                                    all_results.append(RawResult(unique_id=unique_id,
                                                                 start_logits=start_logits,
                                                                 end_logits=end_logits))
                            if mpi_rank == 0:
                                output_prediction_file = os.path.join(args.checkpoint_dir,
                                                                      'prediction_epoch' + str(i) + '.json')
                                output_nbest_file = os.path.join(args.checkpoint_dir, 'nbest_epoch' + str(i) + '.json')

                                write_predictions(dev_examples, dev_data, all_results,
                                                  n_best_size=args.n_best, max_answer_length=args.max_ans_length,
                                                  do_lower_case=True, output_prediction_file=output_prediction_file,
                                                  output_nbest_file=output_nbest_file)
                                tmp_result = get_eval(args.dev_file, output_prediction_file)
                                tmp_result['STEP'] = global_steps
                                print_rank0(tmp_result)
                                with open(args.log_file, 'a') as aw:
                                    aw.write(json.dumps(str(tmp_result)) + '\n')

                                if float(tmp_result['F1']) > best_f1:
                                    best_f1 = float(tmp_result['F1'])
                                if float(tmp_result['EM']) > best_em:
                                    best_em = float(tmp_result['EM'])

                                if float(tmp_result['F1']) + float(tmp_result['EM']) > best_f1_em:
                                    best_f1_em = float(tmp_result['F1']) + float(tmp_result['EM'])
                                    scores = {'F1': float(tmp_result['F1']), 'EM': float(tmp_result['EM'])}
                                    save_prex = "checkpoint_score"
                                    for k in scores:
                                        save_prex += ('_' + k + '-' + str(scores[k])[:6])
                                    save_prex += '.ckpt'
                                    saver.save(get_session(sess),
                                               save_path=os.path.join(args.checkpoint_dir, save_prex))

        F1s.append(best_f1)
        EMs.append(best_em)

    if mpi_rank == 0:
        print('Mean F1:', np.mean(F1s), 'Mean EM:', np.mean(EMs))
        print('Best F1:', np.max(F1s), 'Best EM:', np.max(EMs))
        with open(args.log_file, 'a') as aw:
            aw.write('Mean(Best) F1:{}({})\n'.format(np.mean(F1s), np.max(F1s)))
            aw.write('Mean(Best) EM:{}({})\n'.format(np.mean(EMs), np.max(EMs)))
