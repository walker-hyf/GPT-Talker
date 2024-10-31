# -*- coding: utf-8 -*-
import sys, os
from feature_extractor import cnhubert
import pdb, traceback, numpy as np, logging
from scipy.io import wavfile
import librosa, torch
from time import time as ttime
import shutil
import glob
from my_utils import load_audio
now_dir = os.getcwd()
sys.path.append(now_dir)


def my_save(fea,path): 
    dir = os.path.dirname(path)
    name = os.path.basename(path)
    tmp_path = "%s/%s.pth"%(dir,ttime())
    torch.save(fea, tmp_path)
    shutil.move(tmp_path, "%s/%s"%(dir,name))

def name2go(wav_name,hubert_dir):
    wav_basename = os.path.basename(wav_name)
    hubert_path = "%s/%s.pt"%(hubert_dir,wav_basename)
    if(os.path.exists(hubert_path)):return

    c_dialogue = wav_basename.split("_")[2].strip("d").strip(".wav") 
    wav_path = wav_name
    print(wav_path)
    
    ssl_len = []   

    c_index = wav_basename.split("_")[0]    
    source_folder = "%s/%s/"%(inp_wav_dir,c_dialogue)
    matching_wav_files = []

    ssl = None    

    tmp_audio = load_audio(wav_path, 32000)
    tmp_max = np.abs(tmp_audio).max()

    tmp_audio32 = (tmp_audio / tmp_max * (maxx * alpha*32768)) + ((1 - alpha)*32768) * tmp_audio
    tmp_audio = librosa.resample(
        tmp_audio32, orig_sr=32000, target_sr=16000
    )

    tensor_wav16 = torch.from_numpy(tmp_audio)

    if (is_half == True):
        tensor_wav16=tensor_wav16.half().to(device)
    else:
        tensor_wav16 = tensor_wav16.to(device)

    c_ssl = model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1,2).cpu() 

    if(ssl == None):
        ssl = c_ssl
        ssl_len.append(c_ssl.size(-1))
    else:
        ssl = torch.cat([ssl, c_ssl], dim=-1)
        ssl_len.append(c_ssl.size(-1))

    if np.isnan(ssl.detach().numpy()).sum() != 0:
        print("--")
        return

    wavfile.write(
        "%s/%s"%(wav32dir, wav_basename),
        32000,
        tmp_audio32.astype("int16"),
    )
    my_save(ssl, hubert_path)
    opt.append("%s\t%s"%(wav_basename, ssl_len))


if __name__ == "__main__":
    # //////////////////////////////////////////// 

    source_path = "I:\\GPT-Talker\\"
    exp_name = "DailyTalk"
    h_turn = 2   # Dialogue history rounds

    # ////////////////////////////////////////////
    
    opt_dir = source_path+"\\datasets\\processed\\"+exp_name+"\\"
    inp_text= "%s/slicer_opt.list"%(opt_dir) 
    inp_wav_dir = source_path+"\\datasets\\raw\\"+exp_name
    txt_path = "%s/2-cnhubert-len.txt"%(opt_dir) 
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    bert_pretrained_dir = source_path+"pretrained_models\\chinese-hubert-base\\"
    cnhubert.cnhubert_base_path = bert_pretrained_dir
    is_half = eval(os.environ.get("is_half", "True"))

    hubert_dir="%s/4-cnhubert"%(opt_dir)
    wav32dir="%s/5-wav32k"%(opt_dir)
    os.makedirs(opt_dir,exist_ok=True)
    os.makedirs(hubert_dir,exist_ok=True)
    os.makedirs(wav32dir,exist_ok=True)

    maxx = 0.95
    alpha = 0.5
    device = "cuda:0"
    model = cnhubert.get_model()
    if(is_half==True):
        model=model.half().to(device)
    else:
        model = model.to(device)

    opt=[]

    
    with open(inp_text, "r", encoding="utf8")as f:
        lines = f.read().strip("\n").split("\n")

    for line in lines:
        try:
            wav_name, spk_name, language, text = line.split("|")
            name2go(wav_name,hubert_dir)

        except:
            print(line,traceback.format_exc())

    with open(txt_path, "w", encoding="utf8") as f:
            f.write("\n".join(opt)+"\n")
    