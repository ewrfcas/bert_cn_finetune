import json
from tqdm import tqdm
import collections
import tokenizations.official_tokenization as tokenization
import os
import numpy as np
from preprocess.prepro_utils import *
import gc

SPIECE_UNDERLINE = '▁'


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def json2features(input_file, output_files, tokenizer, is_training=False, repeat_limit=3, max_query_length=64,
                  max_seq_length=512, doc_stride=128):
    with open(input_file, 'r') as f:
        train_data = json.load(f)
        train_data = train_data['data']

    def _is_chinese_char(cp):
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def is_fuhao(c):
        if c == '。' or c == '，' or c == '！' or c == '？' or c == '；' or c == '、' or c == '：' or c == '（' or c == '）' \
                or c == '－' or c == '~' or c == '「' or c == '《' or c == '》' or c == ',' or c == '」' or c == '"' or c == '“' or c == '”' \
                or c == '$' or c == '『' or c == '』' or c == '—' or c == ';' or c == '。' or c == '(' or c == ')' or c == '-' or c == '～' or c == '。' \
                or c == '‘' or c == '’':
            return True
        return False

    def _tokenize_chinese_chars(text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if _is_chinese_char(cp) or is_fuhao(char):
                if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
                    output.append(SPIECE_UNDERLINE)
                output.append(char)
                output.append(SPIECE_UNDERLINE)
            else:
                output.append(char)
        return "".join(output)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == SPIECE_UNDERLINE:
            return True
        return False

    # to examples
    examples = []
    mis_match = 0
    for article in tqdm(train_data):
        for para in article['paragraphs']:
            context = para['context']
            context_chs = _tokenize_chinese_chars(context)
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in context_chs:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                if c != SPIECE_UNDERLINE:
                    char_to_word_offset.append(len(doc_tokens) - 1)

            for qas in para['qas']:
                qid = qas['id']
                ques_text = qas['question']
                ans_text = qas['answers'][0]['text']

                start_position_final = None
                end_position_final = None
                if is_training:
                    count_i = 0
                    start_position = qas['answers'][0]['answer_start']

                    end_position = start_position + len(ans_text) - 1
                    while context[start_position:end_position + 1] != ans_text and count_i < repeat_limit:
                        start_position -= 1
                        end_position -= 1
                        count_i += 1

                    while context[start_position] == " " or context[start_position] == "\t" or \
                            context[start_position] == "\r" or context[start_position] == "\n":
                        start_position += 1

                    start_position_final = char_to_word_offset[start_position]
                    end_position_final = char_to_word_offset[end_position]

                    if doc_tokens[start_position_final] in {"。", "，", "：", ":", ".", ","}:
                        start_position_final += 1

                    actual_text = "".join(doc_tokens[start_position_final:(end_position_final + 1)])
                    cleaned_answer_text = "".join(tokenization.whitespace_tokenize(ans_text))

                    if actual_text != cleaned_answer_text:
                        print(actual_text, 'V.S', cleaned_answer_text)
                        mis_match += 1
                        # ipdb.set_trace()

                examples.append({'doc_tokens': doc_tokens,
                                 'orig_answer_text': ans_text,
                                 'qid': qid,
                                 'question': ques_text,
                                 'answer': ans_text,
                                 'start_position': start_position_final,
                                 'end_position': end_position_final})

    print('examples num:', len(examples))
    print('mis_match:', mis_match)
    os.makedirs('/'.join(output_files[0].split('/')[0:-1]), exist_ok=True)
    json.dump(examples, open(output_files[0], 'w'))

    # to features
    features = []
    unique_id = 1000000000
    for (example_index, example) in enumerate(tqdm(examples)):
        query_tokens = tokenizer.tokenize(example['question'])
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example['doc_tokens']):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_start_position = orig_to_tok_index[example['start_position']]  # 原来token到新token的映射，这是新token的起点
            if example['end_position'] < len(example['doc_tokens']) - 1:
                tok_end_position = orig_to_tok_index[example['end_position'] + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example['orig_answer_text'])

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        doc_spans = []
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                if tok_start_position == -1 and tok_end_position == -1:
                    start_position = 0  # 问题本来没答案，0是[CLS]的位子
                    end_position = 0
                else:  # 如果原本是有答案的，那么去除没有答案的feature
                    out_of_span = False
                    doc_start = doc_span.start  # 映射回原文的起点和终点
                    doc_end = doc_span.start + doc_span.length - 1

                    if not (tok_start_position >= doc_start and tok_end_position <= doc_end):  # 该划窗没答案作为无答案增强
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset

            features.append({'unique_id': unique_id,
                             'example_index': example_index,
                             'doc_span_index': doc_span_index,
                             'tokens': tokens,
                             'token_to_orig_map': token_to_orig_map,
                             'token_is_max_context': token_is_max_context,
                             'input_ids': input_ids,
                             'input_mask': input_mask,
                             'segment_ids': segment_ids,
                             'start_position': start_position,
                             'end_position': end_position})
            unique_id += 1

    print('features num:', len(features))
    json.dump(features, open(output_files[1], 'w'))


def _convert_index(index, pos, M=None, is_start=True):
    if pos >= len(index):
        pos = len(index) - 1
    if index[pos] is not None:
        return index[pos]
    N = len(index)
    rear = pos
    while rear < N - 1 and index[rear] is None:
        rear += 1
    front = pos
    while front > 0 and index[front] is None:
        front -= 1
    assert index[front] is not None or index[rear] is not None
    if index[front] is None:
        if index[rear] >= 1:
            if is_start:
                return 0
            else:
                return index[rear] - 1
        return index[rear]
    if index[rear] is None:
        if M is not None and index[front] < M - 1:
            if is_start:
                return index[front] + 1
            else:
                return M - 1
        return index[front]
    if is_start:
        if index[rear] > index[front] + 1:
            return index[front] + 1
        else:
            return index[rear]
    else:
        if index[rear] > index[front] + 1:
            return index[rear] - 1
        else:
            return index[front]


def json2features_xlnet(input_file, output_files, sp_model, is_training=False, max_query_length=64,
                        max_seq_length=512, doc_stride=128):
    special_symbols = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "<cls>": 3,
        "<sep>": 4,
        "<pad>": 5,
        "<mask>": 6,
        "<eod>": 7,
        "<eop>": 8,
    }
    CLS_ID = special_symbols["<cls>"]
    SEP_ID = special_symbols["<sep>"]
    SEG_ID_P = 0
    SEG_ID_Q = 1
    SEG_ID_CLS = 2
    SEG_ID_PAD = 3

    def read_squad_examples(input_file, is_training):
        """Read a SQuAD json file into a list of SquadExample."""
        with open(input_file, "r") as reader:
            input_data = json.load(reader)["data"]

        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    orig_answer_text = None
                    is_impossible = False

                    if is_training:
                        if "is_impossible" in qa:
                            is_impossible = qa["is_impossible"]
                        else:
                            is_impossible = False
                        if (len(qa["answers"]) != 1) and (not is_impossible):
                            raise ValueError(
                                "For training, each question should have exactly 1 answer.")
                        if not is_impossible:
                            answer = qa["answers"][0]
                            orig_answer_text = answer["text"]
                            start_position = answer["answer_start"]
                        else:
                            start_position = -1
                            orig_answer_text = ""

                    example = {
                        "qas_id": qas_id,
                        "question_text": question_text,
                        "paragraph_text": paragraph_text,
                        "orig_answer_text": orig_answer_text,
                        "start_position": start_position,
                        "is_impossible": is_impossible
                    }
                    examples.append(example)

        return examples

    def convert_examples_to_features(examples, sp_model, max_seq_length,
                                     doc_stride, max_query_length, is_training):
        """Loads a data file into a list of `InputBatch`s."""

        cnt_pos, cnt_neg = 0, 0
        unique_id = 1000000000
        max_N, max_M = 1024, 1024
        features = []
        f = np.zeros((max_N, max_M), dtype=np.float32)

        for (example_index, example) in enumerate(examples):

            if example_index % 100 == 0:
                print('Converting {}/{} pos {} neg {}'.format(
                    example_index, len(examples), cnt_pos, cnt_neg))

            query_tokens = encode_ids(
                sp_model,
                preprocess_text(example['question_text'], lower=True))

            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[0:max_query_length]

            paragraph_text = example['paragraph_text']
            para_tokens = encode_pieces(
                sp_model,
                preprocess_text(example['paragraph_text'], lower=True))

            chartok_to_tok_index = []
            tok_start_to_chartok_index = []
            tok_end_to_chartok_index = []
            char_cnt = 0
            for i, token in enumerate(para_tokens):
                chartok_to_tok_index.extend([i] * len(token))
                tok_start_to_chartok_index.append(char_cnt)
                char_cnt += len(token)
                tok_end_to_chartok_index.append(char_cnt - 1)

            tok_cat_text = ''.join(para_tokens).replace(SPIECE_UNDERLINE, ' ')
            N, M = len(paragraph_text), len(tok_cat_text)

            if N > max_N or M > max_M:
                max_N = max(N, max_N)
                max_M = max(M, max_M)
                f = np.zeros((max_N, max_M), dtype=np.float32)
                gc.collect()

            g = {}

            def _lcs_match(max_dist):
                f.fill(0)
                g.clear()

                ### longest common sub sequence
                # f[i, j] = max(f[i - 1, j], f[i, j - 1], f[i - 1, j - 1] + match(i, j))
                for i in range(N):

                    # note(zhiliny):
                    # unlike standard LCS, this is specifically optimized for the setting
                    # because the mismatch between sentence pieces and original text will
                    # be small
                    for j in range(i - max_dist, i + max_dist):
                        if j >= M or j < 0: continue

                        if i > 0:
                            g[(i, j)] = 0
                            f[i, j] = f[i - 1, j]

                        if j > 0 and f[i, j - 1] > f[i, j]:
                            g[(i, j)] = 1
                            f[i, j] = f[i, j - 1]

                        f_prev = f[i - 1, j - 1] if i > 0 and j > 0 else 0
                        if (preprocess_text(paragraph_text[i], lower=True,
                                            remove_space=False)
                                == tok_cat_text[j]
                                and f_prev + 1 > f[i, j]):
                            g[(i, j)] = 2
                            f[i, j] = f_prev + 1

            max_dist = abs(N - M) + 5
            for _ in range(2):
                _lcs_match(max_dist)
                if f[N - 1, M - 1] > 0.8 * N: break
                max_dist *= 2

            orig_to_chartok_index = [None] * N
            chartok_to_orig_index = [None] * M
            i, j = N - 1, M - 1
            while i >= 0 and j >= 0:
                if (i, j) not in g: break
                if g[(i, j)] == 2:
                    orig_to_chartok_index[i] = j
                    chartok_to_orig_index[j] = i
                    i, j = i - 1, j - 1
                elif g[(i, j)] == 1:
                    j = j - 1
                else:
                    i = i - 1

            if all(v is None for v in orig_to_chartok_index) or f[N - 1, M - 1] < 0.8 * N:
                print('MISMATCH DETECTED!')
                continue

            tok_start_to_orig_index = []
            tok_end_to_orig_index = []
            for i in range(len(para_tokens)):
                start_chartok_pos = tok_start_to_chartok_index[i]
                end_chartok_pos = tok_end_to_chartok_index[i]
                start_orig_pos = _convert_index(chartok_to_orig_index, start_chartok_pos, N, is_start=True)
                end_orig_pos = _convert_index(chartok_to_orig_index, end_chartok_pos, N, is_start=False)

                tok_start_to_orig_index.append(start_orig_pos)
                tok_end_to_orig_index.append(end_orig_pos)

            if not is_training:
                tok_start_position = tok_end_position = None

            if is_training and example['is_impossible']:
                tok_start_position = -1
                tok_end_position = -1

            if is_training and not example['is_impossible']:
                start_position = example['start_position']
                end_position = start_position + len(example['orig_answer_text']) - 1

                start_chartok_pos = _convert_index(orig_to_chartok_index, start_position,
                                                   is_start=True)
                tok_start_position = chartok_to_tok_index[start_chartok_pos]

                end_chartok_pos = _convert_index(orig_to_chartok_index, end_position,
                                                 is_start=False)
                tok_end_position = chartok_to_tok_index[end_chartok_pos]
                assert tok_start_position <= tok_end_position

            def _piece_to_id(x):
                if six.PY2 and isinstance(x, unicode):
                    x = x.encode('utf-8')
                return sp_model.PieceToId(x)

            all_doc_tokens = list(map(_piece_to_id, para_tokens))

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_is_max_context = {}
                segment_ids = []
                p_mask = []

                cur_tok_start_to_orig_index = []
                cur_tok_end_to_orig_index = []

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i

                    cur_tok_start_to_orig_index.append(
                        tok_start_to_orig_index[split_token_index])
                    cur_tok_end_to_orig_index.append(
                        tok_end_to_orig_index[split_token_index])

                    is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(SEG_ID_P)
                    p_mask.append(0)

                paragraph_len = len(tokens)

                tokens.append(SEP_ID)
                segment_ids.append(SEG_ID_P)
                p_mask.append(1)

                # note(zhiliny): we put P before Q
                # because during pretraining, B is always shorter than A
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(SEG_ID_Q)
                    p_mask.append(1)
                tokens.append(SEP_ID)
                segment_ids.append(SEG_ID_Q)
                p_mask.append(1)

                cls_index = len(segment_ids)
                tokens.append(CLS_ID)
                segment_ids.append(SEG_ID_CLS)
                p_mask.append(0)

                input_ids = tokens

                # The mask has 0 for real tokens and 1 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [0] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(1)
                    segment_ids.append(SEG_ID_PAD)
                    p_mask.append(1)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(p_mask) == max_seq_length

                span_is_impossible = example['is_impossible']
                start_position = None
                end_position = None
                if is_training and not span_is_impossible:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (tok_start_position >= doc_start and
                            tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        # continue
                        start_position = 0
                        end_position = 0
                        span_is_impossible = True
                    else:
                        # note(zhiliny): we put P before Q, so doc_offset should be zero.
                        # doc_offset = len(query_tokens) + 2
                        doc_offset = 0
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset

                if is_training and span_is_impossible:
                    start_position = cls_index
                    end_position = cls_index

                if example_index < 20:
                    print("*** Example ***")
                    print("unique_id: %s" % (unique_id))
                    print("example_index: %s" % (example_index))
                    print("doc_span_index: %s" % (doc_span_index))
                    print("tok_start_to_orig_index: %s" % " ".join(
                        [str(x) for x in cur_tok_start_to_orig_index]))
                    print("tok_end_to_orig_index: %s" % " ".join(
                        [str(x) for x in cur_tok_end_to_orig_index]))
                    print("token_is_max_context: %s" % " ".join([
                        "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                    ]))
                    print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

                    if is_training and span_is_impossible:
                        print("impossible example span")

                    if is_training and not span_is_impossible:
                        pieces = [sp_model.IdToPiece(token) for token in
                                  tokens[start_position: (end_position + 1)]]
                        answer_text = sp_model.DecodePieces(pieces)
                        print("start_position: %d" % (start_position))
                        print("end_position: %d" % (end_position))
                        print("answer: %s" % (printable_text(answer_text)))

                        # note(zhiliny): With multi processing,
                        # the example_index is actually the index within the current process
                        # therefore we use example_index=None to avoid being used in the future.
                        # The current code does not use example_index of training data.
                if is_training:
                    feat_example_index = None
                else:
                    feat_example_index = example_index

                feature = {
                    "unique_id": unique_id,
                    "example_index": feat_example_index,
                    "doc_span_index": doc_span_index,
                    "tok_start_to_orig_index": tok_start_to_orig_index,
                    "tok_end_to_orig_index": tok_end_to_orig_index,
                    "token_is_max_context": token_is_max_context,
                    "input_ids": input_ids,
                    "input_mask": input_mask,
                    "p_mask": p_mask,
                    "segment_ids": segment_ids,
                    "paragraph_len": paragraph_len,
                    "cls_index": cls_index,
                    "start_position": start_position,
                    "end_position": end_position,
                    "is_impossible": span_is_impossible
                }

                features.append(feature)

                unique_id += 1
                if span_is_impossible:
                    cnt_neg += 1
                else:
                    cnt_pos += 1

        print("Total number of instances: {} = pos {} neg {}".format(
            cnt_pos + cnt_neg, cnt_pos, cnt_neg))

        return features

    examples = read_squad_examples(input_file, is_training)
    print('examples num:', len(examples))
    os.makedirs('/'.join(output_files[0].split('/')[0:-1]), exist_ok=True)
    json.dump(examples, open(output_files[0], 'w'))

    features = convert_examples_to_features(examples, sp_model, max_seq_length, doc_stride,
                                            max_query_length, is_training)
    print('features num:', len(features))
    json.dump(features, open(output_files[1], 'w'))
