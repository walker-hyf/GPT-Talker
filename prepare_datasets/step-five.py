
import os
import traceback
import numpy as np
import pandas as pd
from text import cleaned_text_to_sequence

if __name__ == "__main__":
    
    # ////////////////////////////////////////////

    source_path = "I:\\GPT-Talker\\"
    exp_name = "DailyTalk"

    # ////////////////////////////////////////////

    opt_dir = source_path+"\\datasets\\processed\\"+exp_name+"\\"
    txt_path="%s/2-name2text.txt"%(opt_dir)
    semantic_path = "%s/6-name2semantic.tsv"%(opt_dir)

    train_list_path = "%s/train.list"%(opt_dir)
    train_len_speaker_dir_path = "%s\\train-len-speaker\\"%(opt_dir)
    train_semantic_phoneme_dir_path = "%s\\train-semantic-phoneme\\"%(opt_dir)

    self_max_sec = 54
    pad_val = 1024
    self_min_ps_ratio = 3
    self_max_ps_ratio = 25
    self_hz = int(os.environ.get("hz","25hz")[:-2])

    self_phoneme_data={}
    with open(txt_path,"r",encoding="utf8") as f:
        lines=f.read().strip("\n").split("\n")

    for line in lines:
        tmp = line.split("\t")
        if( len(tmp) != 7 ):continue
        self_phoneme_data[tmp[0]]=[tmp[1],tmp[2],tmp[3],eval(tmp[4]),eval(tmp[5]),eval(tmp[6])]

    self_phone_len = []  
    self_bert_feature_len = [] 
    self_speaker_list = [] 
    self_semantic_len = []  

    self_semantic_data = pd.read_csv(semantic_path, delimiter='\t', encoding="utf-8")
    semantic_data_len = len(self_semantic_data)
    phoneme_data_len = len(self_phoneme_data.keys())
    print("semantic_data_len:", semantic_data_len)
    print("phoneme_data_len:", phoneme_data_len)

    self_semantic_phoneme = []
    self_item_names = []
    idx = 0
    num_not_in = 0
    num_deleted_bigger = 0
    num_deleted_ps = 0
    for i in range(semantic_data_len):
        item_name = self_semantic_data['item_name'][i]

        try:
            if item_name in self_phoneme_data:
                phoneme, word2ph, text, phone_len, bert_feature_len, speaker_list  = self_phoneme_data[item_name]
            else:
                continue
        except Exception:
            traceback.print_exc()
            num_not_in += 1
            continue
        
        semantic_str = self_semantic_data['semantic_audio'][i]
        semantic_ids = [int(idx) for idx in semantic_str.split(' ')]
        semantic_len = eval(self_semantic_data['semantic_len'][i])

        if(len(semantic_len) != len(phone_len)):
            num_not_in += 1
            print(item_name)
            continue

        phoneme = phoneme.split(' ')
        try:
            phoneme_ids = cleaned_text_to_sequence(phoneme)
        except:
            traceback.print_exc()
            num_not_in += 1
            continue
        if len(phoneme_ids) >self_max_sec * self_hz/2.5:
            num_deleted_ps += 1
            continue

        ps_ratio = len(phoneme_ids) / (len(semantic_ids) / self_hz)

        if ps_ratio > self_max_ps_ratio or ps_ratio < self_min_ps_ratio:
            num_deleted_ps += 1
            print(item_name)
            continue

        self_semantic_phoneme.append((semantic_ids, phoneme_ids))
        idx += 1
        self_item_names.append(item_name)

        self_phone_len.append(phone_len)
        self_bert_feature_len.append(bert_feature_len)
        self_speaker_list.append(speaker_list)
        self_semantic_len.append(semantic_len)

    min_num = 100 
    leng = len(self_semantic_phoneme)

    if(leng<min_num):
        tmp1 = self_semantic_phoneme
        tmp2 = self_item_names
        self_semantic_phoneme=[]
        self_item_names=[]
        for _ in range(max(2,int(min_num/leng))):
            self_semantic_phoneme += tmp1
            self_item_names += tmp2

    if num_not_in > 0:
        print(f"there are {num_not_in} semantic datas not in phoneme datas")
    if num_deleted_bigger > 0:
        print(
            f"deleted {num_deleted_bigger} audios who's duration are bigger than {self_max_sec} seconds"
        )
    if num_deleted_ps > 0:
        print(
            f"deleted {num_deleted_ps} audios who's phoneme/sec are bigger than {self_max_ps_ratio} or smaller than {self_min_ps_ratio}"
        )

    print("dataset.__len__():", len(self_phone_len))

    
    os.makedirs(train_len_speaker_dir_path)
    os.makedirs(train_semantic_phoneme_dir_path)
    train_list = []
    for idx in range(0,len(self_phone_len)):
        speaker_list = [int(x) for x in self_speaker_list[idx]]
        basename = self_item_names[idx]
        train_list.append([self_item_names[idx],len(self_semantic_phoneme[idx][0]),len(self_semantic_phoneme[idx][1])])
        train_semantic_phoneme_dir = {}
        train_semantic_phoneme_dir["semantic"] = self_semantic_phoneme[idx][0]
        train_semantic_phoneme_dir["phoneme"]  = self_semantic_phoneme[idx][1]
        np.save(train_semantic_phoneme_dir_path+"\\"+basename+'.npy', train_semantic_phoneme_dir)

        train_len_speaker_dir = {}
        train_len_speaker_dir["phone_len"] = self_phone_len[idx]
        train_len_speaker_dir["bert_feature_len"] = self_bert_feature_len[idx]
        train_len_speaker_dir["speaker_list"] = speaker_list
        train_len_speaker_dir["semantic_len"] = self_semantic_len[idx]
        np.save(train_len_speaker_dir_path+"\\"+basename+'.npy', train_len_speaker_dir)

    with open(train_list_path, 'w', encoding="utf8") as f:
        for item in train_list:
            f.write(str(item) + '\n')