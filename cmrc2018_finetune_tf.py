import argparse
import numpy as np
import tensorflow as tf
import os
from models.tf_modeling import BertModelMRC, BertConfig
from optimizations.tf_optimization import Optimizer
import json
import utils
from evaluate.cmrc2018_evaluate import get_eval
from evaluate.cmrc2018_output import write_predictions
import random
from tqdm import tqdm
import collections
from tokenizations.offical_tokenization import BertTokenizer
from preprocess.cmrc2018_preprocess import json2features


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
    tf.logging.set_verbosity(tf.logging.INFO)

    parser.add_argument('--gpu_ids', type=str, default='0')

    # training parameter
    parser.add_argument('--train_epochs', type=int, default=2)
    parser.add_argument('--n_batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--loss_scale', type=float, default=2.0 ** 15)
    parser.add_argument('--warmup_iters', type=int, default=0.1)
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
                        default='check_points/pretrain_models/roberta_wwm_ext_base/vocab.txt')

    parser.add_argument('--train_dir', type=str, default='dataset/cmrc2018/train_features_roberta512.json')
    parser.add_argument('--dev_dir1', type=str, default='dataset/cmrc2018/dev_examples_roberta512.json')
    parser.add_argument('--dev_dir2', type=str, default='dataset/cmrc2018/dev_features_roberta512.json')
    parser.add_argument('--train_file', type=str, default='origin_data/cmrc2018/cmrc2018_train.json')
    parser.add_argument('--dev_file', type=str, default='origin_data/cmrc2018/cmrc2018_dev.json')
    parser.add_argument('--bert_config_file', type=str,
                        default='check_points/pretrain_models/roberta_wwm_ext_base/bert_config.json')
    parser.add_argument('--init_restore_dir', type=str,
                        default='check_points/pretrain_models/roberta_wwm_ext_base/bert_model.ckpt')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='check_points/cmrc2018/roberta_wwm_ext_base/')
    parser.add_argument('--setting_file', type=str, default='setting.txt')
    parser.add_argument('--log_file', type=str, default='log.txt')

    # use some global vars for convenience
    args = parser.parse_args()
    args.checkpoint_dir += ('/epoch{}_batch{}_lr{}_warmup{}_anslen{}_tf/'
                            .format(args.train_epochs, args.n_batch, args.lr, args.warmup_iters, args.max_ans_length))
    args = utils.check_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    print_rank0('######## generating data ########')

    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    assert args.vocab_size == len(tokenizer.vocab)
    if not os.path.exists(args.train_dir):
        json2features(args.train_file, [args.train_dir.replace('_features_', '_examples_'),
                                        args.train_dir], tokenizer, is_training=True)

    if not os.path.exists(args.dev_dir1) or not os.path.exists(args.dev_dir2):
        json2features(args.dev_file, [args.dev_dir1, args.dev_dir2], tokenizer, is_training=False)

    train_data = json.load(open(args.train_dir, 'r'))
    dev_examples = json.load(open(args.dev_dir1, 'r'))
    dev_data = json.load(open(args.dev_dir2, 'r'))
    if os.path.exists(args.log_file):
        os.remove(args.log_file)

    steps_per_epoch = len(train_data) // args.n_batch
    eval_steps = int(steps_per_epoch * args.eval_epochs)
    dev_steps_per_epoch = len(dev_data) // args.n_batch
    if len(train_data) % args.n_batch != 0:
        steps_per_epoch += 1
    if len(dev_data) % args.n_batch != 0:
        dev_steps_per_epoch += 1
    total_steps = steps_per_epoch * args.train_epochs
    args.warmup_iters = int(args.warmup_iters * total_steps)

    print('steps per epoch:', steps_per_epoch)
    print('total steps:', total_steps)
    print('warmup steps:', args.warmup_iters)

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
    print('######## init model ########')
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
                             num_warmup_steps=args.warmup_iters,
                             hvd=None,
                             use_fp16=args.float16,
                             loss_count=args.loss_count,
                             clip_norm=args.clip_norm,
                             init_loss_scale=args.loss_scale)

    for seed_ in args.seed:
        best_f1, best_em = 0, 0
        with open(args.log_file, 'a') as aw:
            aw.write('===================================' +
                     'SEED:' + str(seed_)
                     + '===================================' + '\n')
        print('SEED:', seed_)
        # random seed
        np.random.seed(seed_)
        random.seed(seed_)
        tf.set_random_seed(seed_)

        train_gen = data_generator(train_data, args.n_batch, shuffle=True, drop_last=False)
        dev_gen = data_generator(dev_data, args.n_batch, shuffle=False, drop_last=False)

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        utils.show_all_variables()
        utils.init_from_checkpoint(args.init_restore_dir)
        RawResult = collections.namedtuple("RawResult",
                                           ["unique_id", "start_logits", "end_logits"])

        saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=1)
        global_steps = 0
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(args.train_epochs):
                print('Starting epoch %d' % (i + 1))
                total_loss = 0
                iteration = 1
                with tqdm(total=steps_per_epoch, desc='Epoch %d' % (i + 1)) as pbar:
                    for _ in range(steps_per_epoch):
                        batch_data = next(train_gen)
                        feed_data = {input_ids: batch_data['input_ids'],
                                     input_masks: batch_data['input_mask'],
                                     segment_ids: batch_data['segment_ids'],
                                     start_positions: batch_data['start_position'],
                                     end_positions: batch_data['end_position']}
                        loss, _ = sess.run([train_model.train_loss, optimization.train_op], feed_dict=feed_data)
                        total_loss += loss
                        pbar.set_postfix({'loss': '{0:1.5f}'.format(total_loss / (iteration + 1e-5))})
                        pbar.update(1)
                        iteration += 1
                        global_steps += 1

                        if global_steps % eval_steps == 0:
                            print('Evaluating...')
                            all_results = []
                            for i_step in tqdm(range(dev_steps_per_epoch)):
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

                            output_prediction_file = os.path.join(args.checkpoint_dir,
                                                                  'prediction_epoch' + str(i) + '.json')
                            output_nbest_file = os.path.join(args.checkpoint_dir, 'nbest_epoch' + str(i) + '.json')

                            write_predictions(dev_examples, dev_data, all_results,
                                              n_best_size=args.n_best, max_answer_length=args.max_ans_length,
                                              do_lower_case=True, output_prediction_file=output_prediction_file,
                                              output_nbest_file=output_nbest_file)
                            tmp_result = get_eval(args.dev_file, output_prediction_file)
                            tmp_result['STEP'] = global_steps
                            with open(args.log_file, 'a') as aw:
                                aw.write(json.dumps(tmp_result) + '\n')
                            print(tmp_result)

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
                                saver.save(sess, save_path=os.path.join(args.checkpoint_dir, save_prex))

        F1s.append(best_f1)
        EMs.append(best_em)

    print('Mean F1:', np.mean(F1s), 'Mean EM:', np.mean(EMs))
    print('Best F1:', np.max(F1s), 'Best EM:', np.max(EMs))
    with open(args.log_file, 'a') as aw:
        aw.write('Mean(Best) F1:{}({})\n'.format(np.mean(F1s), np.max(F1s)))
        aw.write('Mean(Best) EM:{}({})\n'.format(np.mean(EMs), np.max(EMs)))
