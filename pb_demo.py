import tensorflow as tf

with tf.gfile.FastGFile('model.pb', 'rb') as f:
    intput_graph_def = tf.GraphDef()
    intput_graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as p_graph:
        tf.import_graph_def(intput_graph_def)

input_ids = p_graph.get_tensor_by_name("import/input_ids:0")
input_mask = p_graph.get_tensor_by_name('import/input_mask:0')
segment_ids = p_graph.get_tensor_by_name('import/segment_ids:0')
start_logits = p_graph.get_tensor_by_name('import/start_logits:0')
end_logits = p_graph.get_tensor_by_name('import/end_logits:0')

context = "《战国无双3》是由光荣和ω-force开发的战国无双系列的正统第三续作。本作以三大故事为主轴，\
分别是以武田信玄等人为主的《关东三国志》，织田信长等人为主的《战国三杰》，石田三成等人为主的《关原的年轻武者》，\
丰富游戏内的剧情。此部份专门介绍角色，欲知武器情报、奥义字或擅长攻击类型等，请至战国无双系列1.由于乡里大辅先生因故去世，\
不得不寻找其他声优接手。从猛将传 and Z开始。2.战国无双 编年史的原创男女主角亦有专属声优。\
此模式是任天堂游戏谜之村雨城改编的新增模式。本作中共有20张战场地图（不含村雨城），\
后来发行的猛将传再新增3张战场地图。但游戏内战役数量繁多，部分地图会有兼用的状况，\
战役虚实则是以光荣发行的2本「战国无双3 人物真书」内容为主，以下是相关介绍。\
（注：前方加☆者为猛将传新增关卡及地图。）合并本篇和猛将传的内容，村雨城模式剔除\
，战国史模式可直接游玩。主打两大模式「战史演武」&「争霸演武」。系列作品外传作品"
context = context.replace('”', '"').replace('“', '"')

question = "《战国无双3》是由哪两个公司合作开发的？"
question = question.replace('”', '"').replace('“', '"')

import tokenizations.official_tokenization as tokenization

tokenizer = tokenization.BertTokenizer(vocab_file='check_points/pretrain_models/roberta_wwm_ext_large/vocab.txt',
                                       do_lower_case=True)

question_tokens = tokenizer.tokenize(question)
context_tokens = tokenizer.tokenize(context)
input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + context_tokens + ['[SEP]']
print(len(input_tokens))
input_ids_ = tokenizer.convert_tokens_to_ids(input_tokens)
segment_ids_ = [0] * (2 + len(question_tokens)) + [1] * (1 + len(context_tokens))
input_mask_ = [1] * len(input_tokens)

while len(input_ids_) < 512:
    input_ids_.append(0)
    segment_ids_.append(0)
    input_mask_.append(0)

import numpy as np

input_ids_ = np.array(input_ids_).reshape(1, 512)
segment_ids_ = np.array(segment_ids_).reshape(1, 512)
input_mask_ = np.array(input_mask_).reshape(1, 512)

with tf.Session(graph=p_graph) as sess:
    start_logits_, end_logits_ = sess.run([start_logits, end_logits], feed_dict={input_ids: input_ids_,
                                                                                 segment_ids: segment_ids_,
                                                                                 input_mask: input_mask_})
    st = np.argmax(start_logits_[0, :])
    ed = np.argmax(end_logits_[0, :])
    print('Answer:', "".join(input_tokens[st:ed + 1]))
