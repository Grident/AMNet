
# AMNet

#AS for detailed processing about data, please refer to this [link](https://github.com/zjunlp/HVPNeT/blob/main/README.md).

Code for paper "["An Alignment and Matching Network with Hierarchical Visual Features for Multimodal Named Entity and Relation Extraction"]".


Requirements
==========
To run the codes, you need to install the requirements:
```
pip install -r requirements.txt
```

Data Preprocess & Download
==========
To extract visual object images, we first use the NLTK parser to extract noun phrases from the text and apply the [visual grouding toolkit](https://github.com/zyang-ur/onestage_grounding) to detect objects. Detailed steps are as follows: [link](https://github.com/zjunlp/HVPNeT/blob/main/README.md).

==========
You can download the Twitter2015 data via this [link](https://drive.google.com/file/d/1qAWrV9IaiBadICFb7mAreXy3llao_teZ/view?usp=sharing)
and Twitter2017 data via this [link](https://drive.google.com/file/d/1ogfbn-XEYtk9GpUECq1-IwzINnhKGJqy/view?usp=sharing). Please place them in `data/NER_data`.

You can download the MRE data via [Google Drive](https://drive.google.com/file/d/1q5_5vnHJ8Hik1iLA9f5-6nstcvvntLrS/view?usp=sharing).  Please place it in `data/RE_data`.



The expected structure of files is:

```
HMNeT
 |-- data
 |    |-- NER_data
 |    |    |-- twitter2015  # text data
 |    |    |    |-- train.txt
 |    |    |    |-- valid.txt
 |    |    |    |-- test.txt
 |    |    |    |-- twitter2015_train_dict.pth  # {imgname: [object-image]}
 |    |    |    |-- ...
 |    |    |-- twitter2015_images       # raw image data
 |    |    |-- twitter2015_aux_images   # object image data
 |    |    |-- twitter2017
 |    |    |-- twitter2017_images
 |    |    |-- twitter2017_aux_images
 |    |-- RE_data
 |    |    |-- img_org          # raw image data
 |    |    |-- img_vg           # object image data
 |    |    |-- txt              # text data
 |    |    |-- ours_rel2id.json # relation data
 |-- models	# models
 |    |-- bert_model.py
 |    |-- modeling_bert.py
 |-- modules
 |    |-- metrics.py    # metric
 |    |-- train.py  # trainer
 |-- processor
 |    |-- dataset.py    # processor, dataset
 |-- logs     # code logs
 |-- run.py   # main 
 |-- run_ner_task.sh
 |-- run_re_task.sh
```

Train
==========

## NER Task

The data path and GPU related configuration are in the `run.py`. To train ner model, run this script.

```shell
bash run_twitter15.sh
bash run_twitter17.sh
```

## RE Task

To train re model, run this script.

```shell
bash run_re_task.sh
```

Test
==========
## NER Task

To test ner model, you can use the tained model and set `load_path` to the model path, then run following script :

```shell
python -u run.py \
        --dataset_name="twitter15/twitter17" \
        --bert_name="bert-base-uncased" \
        --seed=1234 \
        --only_test \
        --max_seq=128 \
        --use_prompt \
        --use_contrastive\
        --use_matching\
        --prompt_len=12 \
        --sample_ratio=1.0 \
        --load_path='ckpt/ner/twitter15/17/best_model.pth' 

```

## RE Task

To test re model, you can use the tained model and set `load_path` to the model path, then run following script:

```shell
python -u run.py \
        --dataset_name="MRE" \
        --bert_name="bert-base-uncased" \
        --seed=1234 \
        --only_test \
        --max_seq=80 \
        --use_prompt \
        --use_contrastive\
        --use_matching\
        --prompt_len=12 \
        --sample_ratio=1.0 \
        --load_path='ckpt/re/best_model.pth

```

Acknowledgement
==========

The acquisition of Twitter15 and Twitter17 data refer to the code from [UMT](https://github.com/jefferyYu/UMT/), many thanks.

The acquisition of MNRE data for multimodal relation extraction task refer to the code from [MEGA](https://github.com/thecharm/Mega), many thanks.

This article extends the work of [HVPNeT] and references some code from [HVPNeT](https://github.com/zjunlp/HVPNeT/tree/main), many thaks.

