import argparse
import sentencepiece as spm
import numpy as np
import tensorflow as tf
import os

try:
    # horovod must be import before optimizer!
    import horovod.tensorflow as hvd
except:
    print('Please setup horovod before using multi-gpu!!!')
    hvd = None

from models.xlnet_modeling import get_qa_outputs
from optimizations.tf_optimization import Optimizer
import json
import utils
from evaluate.cmrc2018_evaluate import get_eval
from evaluate.DRCD_output import write_predictions_topk
import random
from tqdm import tqdm
import collections
from preprocess.DRCD_preprocess import json2features_xlnet


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

    parser.add_argument('--gpu_ids', type=str, default='4,5,6,7')

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
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--start_n_top', type=int, default=5)
    parser.add_argument('--end_n_top', type=int, default=5)
    parser.add_argument("--dropatt", type=float, default=0.1, help="Attention dropout rate.")
    parser.add_argument("--clamp_len", type=int, default=-1, help="Clamp length.")

    # Parameter initialization
    parser.add_argument("--use_tpu", type=bool, default=False)
    parser.add_argument("--init", type=str, default="normal", help="Initialization method.")
    parser.add_argument("--init_std", type=float, default=0.02, help="Initialization std when init is normal.")
    parser.add_argument("--init_range", type=float, default=0.1, help="Initialization std when init is uniform.")

    # data dir
    parser.add_argument('--spiece_model_file', type=str,
                        default='check_points/pretrain_models/xlnet_mid/spiece.model')

    parser.add_argument('--train_dir', type=str, default='dataset/DRCD/train_features_xlnet512.json')
    parser.add_argument('--dev_dir1', type=str, default='dataset/DRCD/dev_examples_xlnet512.json')
    parser.add_argument('--dev_dir2', type=str, default='dataset/DRCD/dev_features_xlnet512.json')
    parser.add_argument('--train_file', type=str, default='origin_data/DRCD/DRCD_training.json')
    parser.add_argument('--dev_file', type=str, default='origin_data/DRCD/DRCD_dev.json')
    parser.add_argument('--model_config_path', type=str,
                        default='check_points/pretrain_models/xlnet_mid/xlnet_config.json')
    parser.add_argument('--init_restore_dir', type=str,
                        default='check_points/pretrain_models/xlnet_mid/xlnet_model.ckpt')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='check_points/DRCD/xlnet_mid/')
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
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(args.spiece_model_file)
        # assert args.vocab_size == len(tokenizer.vocab)
        if not os.path.exists(args.train_dir):
            json2features_xlnet(args.train_file, [args.train_dir.replace('_features_', '_examples_'),
                                                  args.train_dir], sp_model, is_training=True)

        if not os.path.exists(args.dev_dir1) or not os.path.exists(args.dev_dir2):
            json2features_xlnet(args.dev_file, [args.dev_dir1, args.dev_dir2], sp_model, is_training=False)

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
        input_tensors = {
            "unique_ids": tf.placeholder(tf.int32, [None, ], "unique_ids"),
            "input_ids": tf.placeholder(tf.int32, [None, args.max_seq_length], "input_ids"),
            "input_mask": tf.placeholder(tf.float16 if args.float16 else tf.float32,
                                         [None, args.max_seq_length], "input_mask"),
            "start_positions": tf.placeholder(tf.int32, shape=[None, ], name='start_positions'),
            "end_positions": tf.placeholder(tf.int32, shape=[None, ], name='end_positions'),
            "is_impossible": tf.placeholder(tf.int32, shape=[None, ], name='is_impossible'),
            "segment_ids": tf.placeholder(tf.int32, [None, args.max_seq_length], "segment_ids"),
            "cls_index": tf.placeholder(tf.int32, [None, ], "cls_index"),
            "p_mask": tf.placeholder(tf.float16 if args.float16 else tf.float32, [None, args.max_seq_length], "p_mask")
        }

    # build the models for training and testing/validation
    print_rank0('######## init model ########')
    train_outputs = get_qa_outputs(args, input_tensors, is_training=True)
    eval_outputs = get_qa_outputs(args, input_tensors, is_training=False)

    # Compute loss
    seq_length = tf.shape(input_tensors["input_ids"])[1]


    def compute_loss(log_probs, positions):
        one_hot_positions = tf.one_hot(positions, depth=seq_length, dtype=tf.float32)
        loss = - tf.reduce_sum(one_hot_positions * log_probs, axis=-1)
        loss = tf.reduce_mean(loss)
        return loss


    start_loss = compute_loss(tf.cast(train_outputs["start_log_probs"], tf.float32), input_tensors["start_positions"])
    end_loss = compute_loss(tf.cast(train_outputs["end_log_probs"], tf.float32), input_tensors["end_positions"])
    train_loss = (start_loss + end_loss) * 0.5

    cls_logits = tf.cast(train_outputs["cls_logits"], tf.float32)
    is_impossible = tf.cast(tf.reshape(input_tensors["is_impossible"], [-1]), tf.float32)
    regression_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=is_impossible, logits=cls_logits)
    regression_loss = tf.reduce_mean(regression_loss)
    # note(zhiliny): by default multiply the loss by 0.5 so that the scale is
    # comparable to start_loss and end_loss
    train_loss += regression_loss * 0.5

    optimization = Optimizer(loss=train_loss,
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
                                           ["unique_id", "start_top_log_probs", "start_top_index",
                                            "end_top_log_probs", "end_top_index", "cls_logits"])

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
                        feed_data = {input_tensors['input_ids']: batch_data['input_ids'],
                                     input_tensors['input_mask']: batch_data['input_mask'],
                                     input_tensors['segment_ids']: batch_data['segment_ids'],
                                     input_tensors['start_positions']: batch_data['start_position'],
                                     input_tensors['end_positions']: batch_data['end_position'],
                                     input_tensors['is_impossible']: batch_data['is_impossible'],
                                     input_tensors['cls_index']: batch_data['cls_index'],
                                     input_tensors['p_mask']: batch_data['p_mask']}
                        loss, _, global_steps, loss_scale = sess.run(
                            [train_loss, optimization.train_op, optimization.global_step,
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
                                feed_data = {input_tensors['input_ids']: batch_data['input_ids'],
                                             input_tensors['input_mask']: batch_data['input_mask'],
                                             input_tensors['segment_ids']: batch_data['segment_ids'],
                                             input_tensors['cls_index']: batch_data['cls_index'],
                                             input_tensors['p_mask']: batch_data['p_mask']}
                                output_results = sess.run(eval_outputs, feed_dict=feed_data)
                                for j in range(len(batch_data['unique_id'])):
                                    unique_id = int(batch_data["unique_id"][j])
                                    start_top_log_probs = (
                                        [float(x) for x in output_results["start_top_log_probs"][j]])
                                    start_top_index = [int(x) for x in output_results["start_top_index"][j]]
                                    end_top_log_probs = ([float(x) for x in output_results["end_top_log_probs"][j]])
                                    end_top_index = [int(x) for x in output_results["end_top_index"][j]]
                                    cls_logits = float(output_results["cls_logits"][j])
                                    all_results.append(RawResult(unique_id=unique_id,
                                                                 start_top_log_probs=start_top_log_probs,
                                                                 start_top_index=start_top_index,
                                                                 end_top_log_probs=end_top_log_probs,
                                                                 end_top_index=end_top_index,
                                                                 cls_logits=cls_logits))
                            if mpi_rank == 0:
                                output_prediction_file = os.path.join(args.checkpoint_dir,
                                                                      'prediction_epoch' + str(i) + '.json')
                                output_nbest_file = os.path.join(args.checkpoint_dir, 'nbest_epoch' + str(i) + '.json')

                                write_predictions_topk(args, dev_examples, dev_data, all_results,
                                                       n_best_size=args.n_best, max_answer_length=args.max_ans_length,
                                                       output_prediction_file=output_prediction_file,
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
