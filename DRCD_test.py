import os
import argparse
import json
import torch
import utils
from glob import glob
from models.pytorch_modeling import BertConfig, BertForQuestionAnswering, ALBertConfig, ALBertForQA
from evaluate.DRCD_output import write_predictions
from evaluate.cmrc2018_evaluate import get_eval
import collections
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from tokenizations import offical_tokenization as tokenization
from preprocess.DRCD_preprocess import json2features


def test(model, args, eval_examples, eval_features, device):
    print("***** Eval *****")
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])
    output_prediction_file = os.path.join(args.checkpoint_dir, "predictions_test.json")
    output_nbest_file = output_prediction_file.replace('predictions', 'nbest')

    all_input_ids = torch.tensor([f['input_ids'] for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    eval_dataloader = DataLoader(eval_data, batch_size=args.n_batch, shuffle=False)

    model.eval()
    all_results = []
    print("Start evaluating")
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)

        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature['unique_id'])
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))

    write_predictions(eval_examples, eval_features, all_results,
                      n_best_size=args.n_best, max_answer_length=args.max_ans_length,
                      do_lower_case=True, output_prediction_file=output_prediction_file,
                      output_nbest_file=output_nbest_file)

    tmp_result = get_eval(args.dev_file, output_prediction_file)
    print(tmp_result)

    model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='6')

    # training parameter
    parser.add_argument('--train_epochs', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--warmup_rate', type=float, default=0.05)
    parser.add_argument("--schedule", default='warmup_linear', type=str, help='schedule')
    parser.add_argument("--weight_decay_rate", default=0.01, type=float, help='weight_decay_rate')
    parser.add_argument('--float16', type=bool, default=True)  # only sm >= 7.0 (tensorcores)
    parser.add_argument('--max_ans_length', type=int, default=50)
    parser.add_argument('--n_best', type=int, default=20)
    parser.add_argument('--eval_epochs', type=float, default=0.5)
    parser.add_argument('--save_best', type=bool, default=True)
    parser.add_argument('--vocab_size', type=int, default=21128)

    # data dir
    parser.add_argument('--test_dir1', type=str,
                        default='dataset/DRCD/test_examples_roberta512.json')
    parser.add_argument('--test_dir2', type=str,
                        default='dataset/DRCD/test_features_roberta512.json')
    parser.add_argument('--test_file', type=str,
                        default='origin_data/DRCD/DRCD_test.json')
    parser.add_argument('--bert_config_file', type=str,
                        default='check_points/pretrain_models/albert_large_zh/albert_config_large.json')
    parser.add_argument('--vocab_file', type=str,
                        default='check_points/pretrain_models/albert_large_zh/vocab.txt')
    parser.add_argument('--init_restore_dir', type=str,
                        default='check_points/DRCD/albert_large_zh/')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='check_points/DRCD/albert_large_zh/')

    # use some global vars for convenience
    args = parser.parse_args()
    args.checkpoint_dir += ('/epoch{}_batch{}_lr{}_warmup{}_anslen{}/'
                            .format(args.train_epochs, args.n_batch, args.lr, args.warmup_rate, args.max_ans_length))
    args.init_restore_dir += ('/epoch{}_batch{}_lr{}_warmup{}_anslen{}/'
                              .format(args.train_epochs, args.n_batch, args.lr, args.warmup_rate, args.max_ans_length))
    args.init_restore_dir = glob(args.init_restore_dir + '*.pth')
    assert len(args.init_restore_dir) == 1
    args.init_restore_dir = args.init_restore_dir[0]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print("device %s n_gpu %d" % (device, n_gpu))
    print("device: {} n_gpu: {} 16-bits training: {}".format(device, n_gpu, args.float16))

    # load the bert setting
    if 'albert' not in args.bert_config_file:
        bert_config = BertConfig.from_json_file(args.bert_config_file)
    else:
        bert_config = ALBertConfig.from_json_file(args.bert_config_file)

    # load data
    print('loading data...')
    tokenizer = tokenization.BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    assert args.vocab_size == len(tokenizer.vocab)

    if not os.path.exists(args.test_dir1) or not os.path.exists(args.test_dir2):
        json2features(args.test_file, [args.test_dir1, args.test_dir2], tokenizer, is_training=False,
                      max_seq_length=bert_config.max_position_embeddings)

    test_examples = json.load(open(args.test_dir1, 'r'))
    test_features = json.load(open(args.test_dir2, 'r'))

    dev_steps_per_epoch = len(test_features) // args.n_batch
    if len(test_features) % args.n_batch != 0:
        dev_steps_per_epoch += 1

    # init model
    print('init model...')
    if 'albert' not in args.init_restore_dir:
        model = BertForQuestionAnswering(bert_config)
    else:
        model = ALBertForQA(bert_config, dropout_rate=args.dropout)
    utils.torch_show_all_params(model)
    utils.torch_init_model(model, args.init_restore_dir)
    if args.float16:
        model.half()
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    test(model, args, test_examples, test_features, device)
