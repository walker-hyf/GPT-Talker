# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/t2s_dataset.py
import pdb
import sys
# sys.path.append("/data/docker/liujing04/gpt-vits/mq-vits-s1bert_no_bert")
import traceback,os
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
import torch,json
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from text import cleaned_text_to_sequence
# from config import exp_dir

def batch_sequences(sequences: List[np.array], axis: int = 0, pad_value: int = 0):
    seq = sequences[0]
    ndim = seq.ndim
    if axis < 0:
        axis += ndim
    dtype = seq.dtype
    pad_value = dtype.type(pad_value)
    seq_lengths = [seq.shape[axis] for seq in sequences]
    max_length = np.max(seq_lengths)

    padded_sequences = []
    for seq, length in zip(sequences, seq_lengths):
        padding = [(0, 0)] * axis + [(0, max_length - length)] + [(0, 0)] * (
                ndim - axis - 1)
        padded_seq = np.pad(
            seq, padding, mode='constant', constant_values=pad_value)
        padded_sequences.append(padded_seq)
    batch = np.stack(padded_sequences)
    return batch

class Text2SemanticDataset(Dataset):
    """dataset class for text tokens to semantic model training."""

    def __init__(self,
                 phoneme_path: str,
                 semantic_path: str,
                 max_sample: int = None,
                 max_sec: int = 100,
                 pad_val: int = 1024,
                 # min value of phoneme/sec
                 min_ps_ratio: int = 3,
                 # max value of phoneme/sec
                 max_ps_ratio: int = 25) -> None:
        super().__init__()

        # print(phoneme_path)   # 2-name2text.txt
        # print(semantic_path)  # 6-name2semantic.tsv
        
        # （The following three items are all time series, conversation history + current sentence）
        self.phone_len = []  # Number of phonemes per sentence
        self.bert_feature_len = [] # The length of BERT feature per sentence
        self.speaker_list = []  # Speaker information for each sentence
        self.semantic_len = []  # The number of semantic tokens per sentence

        self.path2 = phoneme_path  # 2-name2text.txt
        self.path3 = "%s/3-bert/"%(os.path.dirname(phoneme_path)) # bert_dir
        self.path6 = semantic_path # 6-name2semantic.tsv
        train_path = "%s/train.list"%(os.path.dirname(phoneme_path)) # train.list

        self.train_len_speaker_dir_path = "%s/train-len-speaker/"%(os.path.dirname(phoneme_path))  # "logs\\DailyTalk\\train-len-speaker\\"
        self.train_semantic_phoneme_dir_path = "%s/train-semantic-phoneme/"%(os.path.dirname(phoneme_path))  # train-semantic-phoneme

        assert os.path.exists(self.path2)
        assert os.path.exists(self.path6)
        assert os.path.exists(train_path)
        
        with open(train_path, "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")
        
        self.item_names = []
        self.semantic_len = []
        for line in lines:
            basename = eval(line)[0]
            index = int(basename.split("_")[0])
            if(index != 1):   
                continue
            self.item_names.append(eval(line)[0])
            self.semantic_len.append(eval(line)[1])

        self.PAD: int = pad_val
        self.hz=int(os.environ.get("hz","25hz")[:-2])

    def init_batch(self):
        semantic_data_len = len(self.semantic_data)
        phoneme_data_len = len(self.phoneme_data.keys())
        print("semantic_data_len:", semantic_data_len)
        print("phoneme_data_len:", phoneme_data_len)
        idx = 0
        num_not_in = 0
        num_deleted_bigger = 0
        num_deleted_ps = 0
        for i in range(semantic_data_len):
            item_name = self.semantic_data['item_name'][i]
            try:
                phoneme, word2ph, text, phone_len, bert_feature_len, speaker_list  = self.phoneme_data[item_name]
            except Exception:
                traceback.print_exc()
                num_not_in += 1
                continue
            
            semantic_str = self.semantic_data['semantic_audio'][i]
            # get token list
            semantic_ids = [int(idx) for idx in semantic_str.split(' ')]
            semantic_len = eval(self.semantic_data['semantic_len'][i])

            if(len(semantic_len) != len(phone_len)):
                # Some conversations have excluded some semantics that have Nan, 
                # so these conversations are not considered.
                num_not_in += 1
                continue
            phoneme = phoneme.split(' ')

            try:
                phoneme_ids = cleaned_text_to_sequence(phoneme)
            except:
                traceback.print_exc()
                num_not_in += 1
                continue

            if len(phoneme_ids) >self.max_sec * self.hz/2.5:
                num_deleted_ps += 1
                print(self.max_sec)
                continue

            ps_ratio = len(phoneme_ids) / (len(semantic_ids) / self.hz)
       
            if ps_ratio > self.max_ps_ratio or ps_ratio < self.min_ps_ratio:
                num_deleted_ps += 1
                continue

            self.semantic_phoneme.append((semantic_ids, phoneme_ids))
            idx += 1
            self.item_names.append(item_name)
            self.phone_len.append(phone_len)
            self.bert_feature_len.append(bert_feature_len)
            self.speaker_list.append(speaker_list)
            self.semantic_len.append(semantic_len)

        min_num = 100  
        leng = len(self.semantic_phoneme)

        if(leng<min_num):
            tmp1=self.semantic_phoneme
            tmp2=self.item_names
            self.semantic_phoneme=[]
            self.item_names=[]
            for _ in range(max(2,int(min_num/leng))):
                self.semantic_phoneme += tmp1
                self.item_names += tmp2
        if num_not_in > 0:
            print(f"there are {num_not_in} semantic datas not in phoneme datas")
        if num_deleted_bigger > 0:
            print(
                f"deleted {num_deleted_bigger} audios who's duration are bigger than {self.max_sec} seconds"
            )
        if num_deleted_ps > 0:
            print(
                f"deleted {num_deleted_ps} audios who's phoneme/sec are bigger than {self.max_ps_ratio} or smaller than {self.min_ps_ratio}"
            )

        print("dataset.__len__():", self.__len__())

    def __get_item_names__(self) -> List[str]:
        return self.item_names

    def __len__(self) -> int:
        return len(self.item_names)

    def __getitem__(self, idx: int) -> Dict:
        
        item_name = item = self.item_names[idx]

        len_speaker = np.load(self.train_len_speaker_dir_path+item+'.npy', allow_pickle=True).item()
        
        phone_len = len_speaker["phone_len"]
        bert_feature_len = len_speaker["bert_feature_len"]
        speaker_list = len_speaker["speaker_list"]
        semantic_len = len_speaker["semantic_len"]

        semantic_phoneme = np.load(self.train_semantic_phoneme_dir_path+item+'.npy', allow_pickle=True).item()
        semantic_ids = semantic_phoneme["semantic"]
        phoneme_ids = semantic_phoneme["phoneme"]

        phoneme_ids_len = len(phoneme_ids)
        # semantic tokens target
        semantic_ids_len = len(semantic_ids)

        flag=0
        path_bert = "%s/%s.pt" % (self.path3, item_name)
        if(os.path.exists(path_bert)==True):bert_feature = torch.load(path_bert,map_location="cpu")
        else:flag=1
        if(flag==1):
            # bert_feature=torch.zeros_like(phoneme_ids,dtype=torch.float32)
            bert_feature=None
        else:
            assert bert_feature.shape[-1] == len(phoneme_ids)

        return {
            'idx': idx,
            'phoneme_ids': phoneme_ids,
            'phoneme_ids_len': phoneme_ids_len,
            'semantic_ids': semantic_ids,
            'semantic_ids_len': semantic_ids_len,
            'bert_feature': bert_feature,
            'phone_len': phone_len,
            'bert_feature_len': bert_feature_len,
            'speaker_list': speaker_list,
            'semantic_len': semantic_len
        }

    def get_sample_length(self, idx: int):
        # semantic_ids = self.semantic_phoneme[idx][0]
        semantic_ids_len = int(self.semantic_len[idx])
        sec = 1.0 * semantic_ids_len / self.hz
        return sec

    def collate(self, examples: List[Dict]) -> Dict:
        sample_index: List[int] = []
        phoneme_ids: List[torch.Tensor] = []
        phoneme_ids_lens: List[int] = []
        semantic_ids: List[torch.Tensor] = []
        semantic_ids_lens: List[int] = []

        phone_len: List[List] = []
        bert_feature_len: List[List] = []
        speaker_list: List[List] = []
        semantic_len: List[List] = []

        for item in examples:
            phone_len.append(item["phone_len"])
            bert_feature_len.append(item["bert_feature_len"])
            speaker_list.append(item["speaker_list"])
            semantic_len.append(item["semantic_len"])

            sample_index.append(item["idx"])
            phoneme_ids.append(np.array(item["phoneme_ids"], dtype=np.int64))
            semantic_ids.append(np.array(item["semantic_ids"], dtype=np.int64))
            phoneme_ids_lens.append(item["phoneme_ids_len"])
            semantic_ids_lens.append(item["semantic_ids_len"])

        # pad 0
        phoneme_ids = batch_sequences(phoneme_ids)
        semantic_ids = batch_sequences(semantic_ids, pad_value=self.PAD)

        # # convert each batch to torch.tensor
        phoneme_ids = torch.tensor(phoneme_ids)
        semantic_ids = torch.tensor(semantic_ids)
        phoneme_ids_lens = torch.tensor(phoneme_ids_lens)
        semantic_ids_lens = torch.tensor(semantic_ids_lens)
        bert_padded = torch.FloatTensor(len(examples), 1024, max(phoneme_ids_lens))
        bert_padded.zero_()

        for idx, item in enumerate(examples):
            bert = item['bert_feature']
            if(bert!=None):
                bert_padded[idx, :, :bert.shape[-1]] = bert

        return {
            # List[int]
            "ids": sample_index,
            # torch.Tensor (B, max_phoneme_length)
            "phoneme_ids": phoneme_ids,
            # torch.Tensor (B)
            "phoneme_ids_len": phoneme_ids_lens,
            # torch.Tensor (B, max_semantic_ids_length)
            "semantic_ids": semantic_ids,
            # torch.Tensor (B)
            "semantic_ids_len": semantic_ids_lens,
            # torch.Tensor (B, 1024, max_phoneme_length)
            "bert_feature": bert_padded,

            # the number of phonemes in each sentence in the conversation
            "phone_len": phone_len,
            # Bert feature dimensions for each sentence in the conversation (only available in Chinese)
            "bert_feature_len": bert_feature_len,
            # Speaker information for each sentence in the conversation
            "speaker_list": speaker_list,
            "semantic_len": semantic_len
        }


if __name__ == '__main__':

    root_dir = 'I:\\GPT-Talker\\datasets\\processed\\DailyTalk\\'
    dataset = Text2SemanticDataset(
        phoneme_path=root_dir + '2-name2text.txt',
        semantic_path=root_dir + '6-name2semantic.tsv')

    batch_size = 2
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate,
        shuffle=False)
    
    for i, batch in enumerate(dataloader):
        
        if(i%1000==0): print(batch)

        # if i == 0:
        #     print('batch["ids"]:', batch["ids"])
            # print('batch["phoneme_ids"]:', batch["phoneme_ids"],
            #       batch["phoneme_ids"].shape)
            # print('batch["phoneme_ids_len"]:', batch["phoneme_ids_len"],
            #       batch["phoneme_ids_len"].shape)
            # print('batch["semantic_ids"]:', batch["semantic_ids"],
            #       batch["semantic_ids"].shape)
            # print('batch["semantic_ids_len"]:', batch["semantic_ids_len"],
            #       batch["semantic_ids_len"].shape)
