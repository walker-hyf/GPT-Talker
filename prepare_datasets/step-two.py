# -*- coding: utf-8 -*-
import os
import re
import sys, numpy as np, traceback, pdb
import os.path
import glob
from tqdm import tqdm
from text.cleaner import clean_text
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
from time import time as ttime
import shutil



def my_save(fea,path):  #####fix issue: torch.save doesn't support chinese path
    dir=os.path.dirname(path)
    name=os.path.basename(path)
    tmp_path="%s/%s.pth"%(dir,ttime())
    torch.save(fea,tmp_path)
    shutil.move(tmp_path,"%s/%s"%(dir,name))

def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


def process(data,res):
    # Need to get information about the history of the conversation 
    # (bert features of the text and hubert features of the speech) 
    # as each speech is processed here

    for name, text, lan in data:
        try:
            print(name)
            speaker_list = []      # All speaker information for the entire conversation (in chronological order)
            phone_sq = ""          # Phonemes  (in chronological order)
            phone_len = []         # Record the number of phonemes in each sentence (in chronological order)
            bert_feature_sq = None # Bert feature of the entire conversation (in chronological order)
            bert_feature_len = []  # Length of bert features per utterance (in chronological order)
            word2ph = []           # Number of phonemes/phonetic symbols per sentence, per word

            source_folder = os.path.dirname(name)
            name = os.path.basename(name)
            print(name)   # 0_1_d0.wav
            c_index = name.split("_")[0]
            c_dialogue = name.split("_")[2].strip("d").strip(".wav")

            # Getting information about dialog history
            matching_wav_files = []
            matching_txt_files = []
            current_index = 0
            norm_text = ""
            bert_feature = None

            
            for current_index in range(0, int(c_index)):
                # Get all wavs less than index
                if(current_index<int(c_index) and current_index>=(int(c_index)-h_turn)):
                    print(current_index)
                    print(c_dialogue)
                    wav_file = glob.glob(source_folder+"\\"+str(current_index)+"_*_d"+c_dialogue+".wav")[0]
                    matching_wav_files.append(wav_file)
                    
                    txt_file = glob.glob(source_folder+"\\"+str(current_index)+"_*_d"+c_dialogue+".txt")[0]
                    if not txt_file:
                        txt_file = glob.glob(source_folder+"\\"+str(current_index)+"_*_d"+c_dialogue+".lab")[0]
                    matching_txt_files.append(txt_file)
                    
                    # Get the phoneme of each sentence
                    with open(txt_file, "r", encoding="utf8") as file:
                        content = file.read()
                        h_phones, h_word2ph, h_norm_text = clean_text(content.replace("%", '-').replace('￥', ','),lan)

                        h_phone_len = len(h_phones)  
                        phone_len.append(h_phone_len)
                        h_phones = " ".join(h_phones)   
                        phone_sq += h_phones+" "
                        norm_text += h_norm_text+" "

                        if (lan == "zh"):
                            h_bert_feature = get_bert_feature(h_norm_text, h_word2ph)
                            assert h_bert_feature.shape[-1] == len(h_phones.split(" "))
                            word2ph += h_word2ph

                            if(bert_feature == None):
                                bert_feature = h_bert_feature
                                bert_feature_len.append(h_bert_feature.shape[-1])
                            else:
                                bert_feature = torch.cat([bert_feature, h_bert_feature], dim=-1)
                                bert_feature_len.append(h_bert_feature.shape[-1])
                    
                    h_speaker = os.path.basename(wav_file).split("_")[1]
                    speaker_list.append(h_speaker)


            # Add the contents of the current utterance
            c_speaker = name.split("_")[1]
            speaker_list.append(c_speaker)

            c_phones, c_word2ph, c_norm_text = clean_text(text.replace("%", '-').replace('￥', ','),lan)
            path_bert="%s/%s.pt"%(bert_dir,name)

            if (os.path.exists(path_bert) == False and lan == "zh"):
                c_bert_feature = get_bert_feature(c_norm_text, c_word2ph)
                assert c_bert_feature.shape[-1] == len(c_phones)
                word2ph += c_word2ph
                
                if(bert_feature == None):
                    bert_feature = c_bert_feature
                    bert_feature_len.append(c_bert_feature.shape[-1])
                else:
                    bert_feature = torch.cat([bert_feature, c_bert_feature], dim=-1)
                    bert_feature_len.append(c_bert_feature.shape[-1])

                my_save(bert_feature, path_bert)

            c_phone_len = len(c_phones)    
            phone_len.append(c_phone_len)
            c_phones = " ".join(c_phones)   
            phone_sq += c_phones
            norm_text += c_norm_text
            res.append([name, phone_sq, word2ph, norm_text, phone_len, bert_feature_len, speaker_list])

        except:
            print(name, text, traceback.format_exc())


if __name__ == "__main__":

    # //////////////////////////////////////////// 

    source_path = "I:\\GPT-Talker\\"
    exp_name = "DailyTalk"
    h_turn = 2   # Dialogue history rounds

    # ////////////////////////////////////////////

    inp_text = source_path+"\\datasets\\processed\\"+exp_name+"\\slicer_opt.list"
    inp_wav_dir = source_path+"\\datasets\\raw\\"+exp_name
    opt_dir = source_path+"\\datasets\\processed\\"+exp_name+"\\"
    bert_pretrained_dir = source_path+"pretrained_models\\chinese-roberta-wwm-ext-large\\"
    txt_path="%s/2-name2text.txt"%(opt_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    is_half = eval(os.environ.get("is_half", "True"))
    device="cuda:0"

    if(os.path.exists(txt_path)==False):
        bert_dir="%s/3-bert"%(opt_dir)
        os.makedirs(opt_dir,exist_ok=True)
        os.makedirs(bert_dir,exist_ok=True)
        
        tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_dir)
        bert_model= AutoModelForMaskedLM.from_pretrained(bert_pretrained_dir)

        if (is_half == True):
            bert_model = bert_model.half().to(device)
        else:
            bert_model = bert_model.to(device)

        todo=[]
        res=[]
        with open(inp_text,"r",encoding="utf8") as f:
            lines=f.read().strip("\n").split("\n")

        language_v1_to_language_v2={
            "ZH":"zh",
            "EN":"en"
        }
        for line in lines:
            try:
                wav_name,spk_name,language,text=line.split("|")
                # todo.append([name,text,"zh"])
                todo.append([wav_name,text,language_v1_to_language_v2.get(language,language)])
            except:
                print(line,traceback.format_exc())

        process(todo, res)

        opt=[]
        for name, phones, word2ph, norm_text, phone_len, bert_feature_len, speaker_list  in res:
            opt.append("%s\t%s\t%s\t%s\t%s\t%s\t%s"%(name, phones, word2ph, norm_text, phone_len, bert_feature_len, speaker_list))


        with open(txt_path,"w",encoding="utf8") as f:
            f.write("\n".join(opt)+"\n")