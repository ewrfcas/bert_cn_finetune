## BERT下游任务finetune列表

finetune基于官方代码改造的模型基于pytorch/tensorflow双版本

*** 2019-10-24: 增加ERNIE1.0, google-bert-base, bert_wwm_ext_base部分结果, xlnet代码和相关结果 ***

*** 2019-10-17: 增加tensorflow多gpu并行 ***

*** 2019-10-16: 增加albert_xlarge结果 ***

*** 2019-10-15: 增加tensorflow(bert/roberta)在cmrc2018上的finetune代码(暂仅支持单卡) ***

*** 2019-10-14: 新增DRCD test结果 ***

*** 2019-10-12: pytorch支持albert ***

*** 2019-12-9: 新增cmrc2019 finetune google版albert, 新增CHID finetune代码***

*** 2019-12-22: 新增c3 finetune代码***

### 模型及相关代码来源

1. 官方Bert (https://github.com/google-research/bert)

2. transformers (https://github.com/huggingface/transformers)

3. 哈工大讯飞预训练 (https://github.com/ymcui/Chinese-BERT-wwm)

4. brightmart预训练 (https://github.com/brightmart/roberta_zh)

5. 自己瞎折腾的siBert (https://github.com/ewrfcas/SiBert_tensorflow)

### 关于pytorch的FP16

FP16的训练可以显著降低显存压力(如果有V100等GPU资源还能提高速度)。但是最新版编译的apex-FP16对并行的支持并不友好(https://github.com/NVIDIA/apex/issues/227)  
实践下来bert相关任务的finetune任务对fp16的数值压力是比较小的，因此可以更多的以计算精度换取效率，所以我还是倾向于使用老版的FusedAdam+FP16_Optimizer的组合。  
由于最新的apex已经舍弃这2个方法了，需要在编译apex的时候额外加入命令--deprecated_fused_adam  
```
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"  --global-option="--deprecated_fused_adam" ./
```

### 关于tensorflow的blocksparse

blocksparse(https://github.com/openai/blocksparse)  
可以在tensorflow1.13版本直接pip安装，否则可以自己clone后编译。  
其中fast_gelu以及self-attention中的softmax能够极大缓解显存压力。另外部分dropout位置我有所调整，整体显存占用下降大约30%~40%。

model | length | batch | memory |
| ------ | ------ | ------ | ------ |
| roberta_base_fp16 | 512 | 32 | 16GB |
| roberta_large_fp16 | 512 | 12 | 16GB |


### 参与任务

1. CMRC 2018：篇章片段抽取型阅读理解（简体中文，只测了dev）

2. DRCD：篇章片段抽取型阅读理解（繁体中文，转简体, 只测了dev）

3. CJRC: 法律阅读理解（简体中文, 只有训练集，统一90%训练，10%测试）

4. XNLI：自然语言推断 (todo)

5. Msra-ner：中文命名实体识别 (todo)

6. THUCNews：篇章级文本分类 (todo)

7. Dbqa: ...

8. Chnsenticorp: ...

9. Lcqmc: ...

### 评测标准

验证集一般会调整learning_rate，warmup_rate，train_epoch等参数，选择最优的参数用五个不同的随机种子测试5次取平均和括号内最大值。测试集会直接用最佳的验证集模型进行验证。

### 模型介绍

L(transformer layers), H(hidden size), A(attention head numbers), E(embedding size)

**特别注意brightmart roberta_large所支持的max_len只有256**

| models | config |
| ------ | ------ |
| google_bert_base | L=12, H=768, A=12, max_len=512 |
| siBert_base | L=12, H=768, A=12, max_len=512 |
| siALBert_middle | L=16, H=1024, E=128, A=16, max_len=512 |
| 哈工大讯飞 bert_wwm_ext_base | L=12, H=768, A=12, max_len=512 |
| 哈工大讯飞 roberta_wwm_ext_base | L=12, H=768, A=12, max_len=512 |
| 哈工大讯飞 roberta_wwm_ext_large | L=24, H=1024, A=16, max_len=512 |
| ERNIE1.0 | L=12, H=768, A=12, max_len=512 |
| xlnet-mid | L=24, H=768, A=12, max_len=512 |
| brightmart roberta_middle | L=24, H=768, A=12, max_len=512 |
| brightmart roberta_large | L=24, H=1024, A=16, **max_len=256** |
| brightmart albert_large | L=24, H=1024, E=128, A=16, max_len=512 |
| brightmart albert_xlarge | L=24, H=2048, E=128, A=32, max_len=512 |


### 结果

#### 参数

未列出均为epoch2, batch=32, lr=3e-5, warmup=0.1

| models | cmrc2018 | DRCD | CJRC |
| ------ | ------ | ------ | ------ |
| 哈工大讯飞 roberta_wwm_ext_base | epoch2, batch=32, lr=3e-5, warmup=0.1 | 同左 | 同左 |
| 哈工大讯飞 roberta_wwm_ext_large | epoch2, batch=12, lr=2e-5, warmup=0.1 | epoch2, batch=32, lr=2.5e-5, warmup=0.1 | - |
| brightmart roberta_middle | epoch2, batch=32, lr=3e-5, warmup=0.1 | 同左 | 同左 |
| brightmart roberta_large | epoch2, batch=32, lr=3e-5, warmup=0.1 | 同左 | 同左 |
| brightmart albert_large |  epoch3, batch=32, lr=2e-5, warmup=0.05 | epoch3, batch=32, lr=2e-5, warmup=0.05 | epoch2, batch=32, lr=3e-5, warmup=0.1 |
| brightmart albert_xlarge |  epoch3, batch=32, lr=2e-5, warmup=0.1 | epoch3, batch=32, lr=2.5e-5, warmup=0.06 | epoch2, batch=32, lr=2.5e-5, warmup=0.05 |

#### cmrc2018(阅读理解)

| models | setting | DEV |
| ------ | ------ | ------ |
| 哈工大讯飞 roberta_wwm_ext_large | tf单卡finetune batch=12 | **F1:89.415(89.724) EM:70.593(71.358)** |


| models | DEV |
| ------ | ------ |
| google_bert_base | F1:85.476(85.682) EM:64.765(65.921) |
| sibert_base | F1:87.521(88.628) EM:67.381(69.152) |
| sialbert_middle | F1:87.6956(87.878) EM:67.897(68.624) |
| 哈工大讯飞 bert_wwm_ext_base | F1:86.679(87.473) EM:66.959(69.09) |
| 哈工大讯飞 roberta_wwm_ext_base | F1:87.521(88.628) EM:67.381(69.152) |
| 哈工大讯飞 roberta_wwm_ext_large | **F1:89.415(89.724) EM:70.593(71.358)** |
| ERNIE1.0 | F1:87.300(87.733) EM:66.890(68.251) |
| xlnet-mid | F1:85.625(86.076) EM:65.312(66.076) |
| brightmart roberta_middle | F1:86.841(87.242) EM:67.195(68.313) |
| brightmart roberta_large | F1:88.608(89.431) EM:69.935(72.538) |
| brightmart albert_large | F1:87.860(88.43) EM:67.754(69.028) |
| brightmart albert_xlarge | F1:88.657(89.426) EM:68.897(70.643) |

#### DRCD(阅读理解)

| models | DEV | TEST |
| ------ | ------ | ------ |
| google_bert_base | F1:92.296(92.565) EM:86.600(87.089) | F1:91.464 EM:85.485 |
| siBert_base | F1:93.343(93.524) EM:87.968(88.28) | F1:92.818 EM:86.745 |
| siALBert_middle | F1:93.865(93.975) EM:88.723(88.961) | F1:93.857 EM:88.033 |
| 哈工大讯飞 bert_wwm_ext_base | F1:93.265(93.393) EM:88.002(88.28) | F1:92.633 EM:87.145 |
| 哈工大讯飞 roberta_wwm_ext_base | F1:94.257(94.48) EM:89.291(89.642) | F1:93.526 EM:88.119 |
| 哈工大讯飞 roberta_wwm_ext_large | **F1:95.323(95.54) EM:90.539(90.692)** | **F1:95.060 EM:90.696** |
| ERNIE1.0 | F1:92.779(93.021) EM:86.845(87.259) | F1:92.011 EM:86.029 |
| xlnet-mid | F1:92.081(92.175) EM:84.404(84.563) | F1:91.439 EM:83.281 |
| brightmart roberta_large | F1:94.933(95.057) EM:90.113(90.238) | F1:94.254 EM:89.350 |
| brightmart albert_large | F1:93.903(94.034) EM:88.882(89.132) | F1:93.057 EM:87.518 |
| brightmart albert_xlarge | F1:94.626(95.101) EM:89.682(90.125) | F1:94.697 EM:89.780 |

#### CJRC(带有yes,no,unkown的阅读理解)

| models | DEV |
| ------ | ------ |
| siBert_base | F1:80.714(81.14) EM:64.44(65.04) |
| siALBert_middle | F1:80.9838(81.299) EM:63.796(64.202) |
| 哈工大讯飞 roberta_wwm_ext_base | F1:81.510(81.684) EM:64.924(65.574) |
| brightmart roberta_large | F1:80.16(80.475) EM:65.249(66.133) |
| brightmart albert_large | F1:81.113(81.563) EM:65.346(65.727) |
| brightmart albert_xlarge | **F1:81.879(82.328) EM:66.164(66.387)** |


