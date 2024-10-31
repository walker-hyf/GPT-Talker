
import glob
import os
import shutil
import soundfile
import gradio as gr
from transformers import AutoModelForMaskedLM, AutoTokenizer, RobertaTokenizer, RobertaModel
import numpy as np
import librosa, torch
from feature_extractor import cnhubert
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from time import time as ttime
from module.mel_processing import spectrogram_torch
from tools.my_utils import load_audio, DictToAttrRecursive, split
from AR.utils.io import load_yaml_config


test_data_dir = "I:\\GPT-Talker\\demo\\0.NCSSD-ZH-GT\\"
output_dir = "I:\\GPT-Talker\\valid_data\\pred_speech\\"

congpt_path  = os.environ.get("congpt_path", "I:\\GPT-Talker\\GPT_weights\\ZH-e120.ckpt")
convits_path = os.environ.get("convits_path", "I:\\GPT-Talker\\ConVITS_weights\\ZH_e8_s80.pth")

prompt_language = "中文"  
text_language = "中文"  

result = []

dict_language = {"中文": "zh", "英文": "en"}

cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "pretrained_models/chinese-hubert-base"
)
cnhubert.cnhubert_base_path = cnhubert_base_path


if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True"))
device = "cuda"

# ///////////////// roberta ////////////////////////
bert_path = os.environ.get(
    "bert_path", "pretrained_models/chinese-roberta-wwm-ext-large"
)
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)

if is_half == True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)

# ////////////////  load config ////////////////////
dict_s2 = torch.load(convits_path, map_location="cpu")
hps = dict_s2["config"]
hps = DictToAttrRecursive(hps)
hps.model.semantic_frame_rate = "25hz"
dict_s1 = torch.load(congpt_path, map_location="cpu")

config = load_yaml_config("configs/s1longer.yaml")
ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)


# ////////////////  convits  ////////////////////
vq_model = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model
)
if is_half == True:
    vq_model = vq_model.half().to(device)
else:
    vq_model = vq_model.to(device)
vq_model.eval()
print(vq_model.load_state_dict(dict_s2["weight"], strict=False))

hz = 50
max_sec = config["data"]["max_sec"]

# ///////////////////    congpt    /////////////////////
t2s_model = Text2SemanticLightningModule(config, "ojbk", is_train=False)
t2s_model.load_state_dict(dict_s1["weight"])
if is_half == True:
    t2s_model = t2s_model.half()
t2s_model = t2s_model.to(device)
t2s_model.eval()
total = sum([param.nelement() for param in t2s_model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))


# //////////////// roberta feature /////////////////
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


# /////////////////  get spectrogram  //////////////
def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


# /////////////////  generate speech ///////////////
def get_tts_wav(history_wav, history_txt, prompt_language, text, text_language, output_dir, output_name):

    prompt_language, text = prompt_language, text.strip("\n")
    
    history_len = []
    history_phone_id = []
    history_bert = []
    history_semantic = []

    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]

    refer_wav = None
    h_index = 0  # dialogue turns

    # dialogue history
    for h_txt in history_txt:
        phones1, word2ph1, norm_text1 = clean_text(h_txt, prompt_language)
        phones1 = cleaned_text_to_sequence(phones1)
        phones1_len = len(phones1)
        phones1 = torch.LongTensor(phones1).to(device).unsqueeze(0)
        history_phone_id.append(phones1)
        history_len.append(phones1_len)

        if prompt_language == "zh":
            bert1 = get_bert_feature(norm_text1, word2ph1).to(device)
        else:
            bert1 = torch.zeros(
                (1024, len(phones1)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)

        history_bert.append(bert1.to(device).unsqueeze(0))

        with torch.no_grad():
            wav16k, sr = librosa.load(history_wav[h_index], sr=16000)
            
            wav16k = torch.from_numpy(wav16k)
            if is_half == True:
                wav16k = wav16k.half().to(device)
            else:
                wav16k = wav16k.to(device)

            if(h_index == 0):
                refer_wav = history_wav[h_index]
            else:
                h_index += 1

            ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(
                1, 2
            )  # .float()

            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0].unsqueeze(0).to(device)
            history_semantic.append(prompt_semantic)

    # current utterance
    texts = text.split("\n")
    audio_opt = []
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )
    
    for text in texts:
        phones2, word2ph2, norm_text2 = clean_text(text, text_language)
        current_phones = cleaned_text_to_sequence(phones2)
        current_phones = torch.LongTensor(current_phones).to(device).unsqueeze(0)
        
        if text_language == "zh":
            bert2 = get_bert_feature(norm_text2, word2ph2).to(device)
        else:
            bert2 = torch.zeros((1024, len(phones2))).to(bert1)

        current_bert = bert2.to(device).unsqueeze(0)
        
        t2 = ttime()
        current_len = len(phones2)
        with torch.no_grad():
            pred_semantic, idx = t2s_model.model.infer_new(
                history_phone_id,
                history_len,
                history_bert,
                history_semantic,
                current_phones,
                current_bert,
                current_len,
                # prompt_phone_len=ph_offset,
                top_k=config["inference"]["top_k"],
                early_stop_num=hz * max_sec,
            )

        pred_semantic = pred_semantic[:, -idx:].unsqueeze(
            0
        )

        refer = get_spepc(hps, refer_wav)

        if is_half == True:
            refer = refer.half().to(device)
        else:
            refer = refer.to(device)
        
        #  traget speech
        audio = (
            vq_model.decode(
                pred_semantic, current_phones, refer
            )
            .detach()
            .cpu()
            .numpy()[0, 0]
        )
        audio_opt.append(audio)
        audio_opt.append(zero_wav)

    sampling_rate = hps.data.sampling_rate
    audio_data = (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)
    soundfile.write(output_dir + '/'+str(output_name)+'.wav', audio_data, sampling_rate)


if __name__ == "__main__":

    # dialogue turn = 2
    for root, dirs, files in os.walk(test_data_dir):
        for dir in dirs:
            for sub_root, sub_dirs, sub_files in os.walk(os.path.join(root,dir)):
                result = []
                for file in sub_files:
                    file_path = os.path.join(root, file)
                    if file_path.endswith(".wav") or file_path.endswith(".lab"):
                        result.append(file_path)

                files_len = len(result)
                dialog_len = files_len/2 

                for i in range(2,int(dialog_len)):

                    history_wav_0 = glob.glob(sub_root + "/"+str(i-2)+"_*.wav", recursive=True)[0]
                    history_wav_1 = glob.glob(sub_root + "/"+str(i-1)+"_*.wav", recursive=True)[0]

                    history_text_0_path = glob.glob(sub_root + "/"+str(i-2)+"_*.lab", recursive=True)[0]
                    with open(history_text_0_path, 'r', encoding="utf8") as file_0:
                        content_0 = file_0.read()
                    file_0.close()
                    history_text_0 = content_0.strip("\n").strip(" ")

                    history_text_1_path = glob.glob(sub_root + "/"+str(i-1)+"_*.lab", recursive=True)[0]
                    with open(history_text_1_path, 'r', encoding="utf8") as file_1:
                        content_1 = file_1.read()
                    file_1.close()
                    history_text_1 = content_1.strip("\n").strip(" ")

                    current_text_1_path = glob.glob(sub_root + "/"+str(i)+"_*.lab", recursive=True)[0]
                    with open(current_text_1_path, 'r', encoding="utf8") as file_2:
                        content_2 = file_2.read()
                    file_2.close()
                    current_text = content_2.strip("\n").strip(" ")

                    current_file_name = current_text_1_path
                    current_dialog = os.path.basename(current_file_name).strip(".lab").split("_")[-1].strip("d")
                    c_output_dir = os.path.join(output_dir,current_dialog)
                    if not os.path.exists(c_output_dir):
                        os.makedirs(c_output_dir)
                        print(f"Dir: {c_output_dir} Created Successfully") 
                    
                    destination_file = os.path.join(c_output_dir, os.path.basename(current_text_1_path))
                    shutil.copy(current_file_name, destination_file)
                    output_name = "pred-"+os.path.basename(current_text_1_path).strip(".lab")

                    history_txt = [history_text_0, history_text_1]
                    history_wav = [history_wav_0, history_wav_1]

                    if os.path.exists(os.path.join(c_output_dir,output_name+".wav")):
                        continue

                    get_tts_wav(history_wav, history_txt, prompt_language, current_text, text_language, c_output_dir, output_name)
