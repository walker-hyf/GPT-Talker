import os
import math,traceback
import multiprocessing
import sys,pdb
now_dir = os.getcwd()
sys.path.append(now_dir)
from random import shuffle
import torch.multiprocessing as mp
import glob
from tqdm import tqdm
import logging,librosa,utils,torch
from module.models import SynthesizerTrn
from my_utils import load_audio


logging.getLogger("numba").setLevel(logging.WARNING)


if __name__ == "__main__":

    # ////////////////////////////////////////////
     
    source_path = "I:\\GPT-Talker\\"
    exp_name = "DailyTalk"
    h_turn = 2   # Dialogue history rounds

    # //////////////////////////////////////////// 

    inp_text = source_path+"\\datasets\\processed\\"+exp_name+"\\slicer_opt.list"
    inp_wav_dir = source_path+"\\datasets\\raw\\"+exp_name
    opt_dir = source_path+"\\datasets\\processed\\"+exp_name+"\\"
    bert_pretrained_dir = source_path+"pretrained_models\\chinese-hubert-base\\"
    txt_path="%s/2-name2text.txt"%(opt_dir)
    hubert_dir="%s/4-cnhubert"%(opt_dir)
    wav32dir="%s/5-wav32k"%(opt_dir)
    semantic_path = "%s/6-name2semantic.tsv"%(opt_dir)
    
    pretrained_s2G = "pretrained_models\\s2G488k.pth"
    s2config_path = "configs\\s2.json"
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    is_half = eval(os.environ.get("is_half", "True"))


    if(os.path.exists(semantic_path)==False):
        os.makedirs(opt_dir, exist_ok=True)

        device="cuda:0"
        hps = utils.get_hparams_from_file(s2config_path)
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model)
        
        if(is_half==True):
            vq_model=vq_model.half().to(device)
        else:
            vq_model = vq_model.to(device)

        vq_model.eval()

        print(vq_model.load_state_dict(torch.load(pretrained_s2G,map_location="cpu")["weight"], strict=False))

        def name2go(wav_name, lines):
            semantic = ""    
            c_dialogue = wav_name.split("_")[2].strip("d").strip(".wav") 
            source_folder = "%s/%s/"%(inp_wav_dir,c_dialogue)
            semantic_len = []                 
            c_index = wav_name.split("_")[0]  
            matching_wav_files = []
            for current_index in range(0, int(c_index)):
                
                if(current_index < int(c_index) and current_index>=(int(c_index) - h_turn)): 
                    
                    h_hubert_paths = glob.glob(hubert_dir+"\\"+str(current_index)+"_*_d"+c_dialogue+".wav.pt")
                    if(len(h_hubert_paths) == 0):
                        continue   
                    else:
                        h_hubert_path = h_hubert_paths[0]
                        if(os.path.exists(h_hubert_path) == False):return

                        h_ssl_content = torch.load(h_hubert_path, map_location="cpu")

                        if(is_half == True):
                            h_ssl_content = h_ssl_content.half().to(device)
                        else:
                            h_ssl_content = h_ssl_content.to(device)

                        h_codes = vq_model.extract_latent(h_ssl_content)

                        if(semantic == ""):
                            semantic = " ".join([str(i) for i in h_codes[0, 0, :].tolist()])
                            semantic = semantic + " "
                            semantic_len.append(len(h_codes[0, 0, :].tolist()))
                        else:
                            semantic += " ".join([str(i) for i in h_codes[0, 0, :].tolist()])
                            semantic = semantic + " "
                            semantic_len.append(len(h_codes[0, 0, :].tolist()))

            c_hubert_path = "%s/%s.pt" % (hubert_dir, wav_name)
            if(os.path.exists(c_hubert_path) == False):return

            c_ssl_content = torch.load(c_hubert_path, map_location="cpu")

            if(is_half == True):
                c_ssl_content = c_ssl_content.half().to(device)
            else:
                c_ssl_content = c_ssl_content.to(device)

            c_codes = vq_model.extract_latent(c_ssl_content)

            if(semantic == ""):
                semantic = " ".join([str(i) for i in c_codes[0, 0, :].tolist()])
                semantic_len.append(len(c_codes[0, 0, :].tolist()))
            else:                           
                semantic += " ".join([str(i) for i in c_codes[0, 0, :].tolist()])
                semantic_len.append(len(c_codes[0, 0, :].tolist()))
            
            lines.append("%s\t%s\t%s"%(wav_name, semantic, semantic_len))

        with open(inp_text,"r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")

        lines1 = []
        count = 0
        for line in lines:
            print(line)
            try:
                wav_name, spk_name, language, text = line.split("|")
                wav_name = os.path.basename(wav_name)
                name2go(wav_name, lines1)
            except:
                print(line, traceback.format_exc())

        header = "item_name\tsemantic_audio\tsemantic_len\n"
        with open(semantic_path, "w", encoding="utf8") as f:
            f.write(header+"\n".join(lines1))