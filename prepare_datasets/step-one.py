import glob
import os
import shutil
import wave

def write_to_list_file(file_path, content_list):
    parent_dir = os.path.dirname(file_path)
    
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        
    with open(file_path, 'w', encoding="utf8") as file:
        for line in content_list:
            file.write(line + '\n')

def get_all_folders(path):
    folder_paths = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            folder_path = os.path.join(root, dir)
            folder_paths.append(folder_path)
    return folder_paths

def get_all_wav_in_dir(dir_path):
    wav_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    return wav_files

if __name__ == "__main__":
    
    # /////////////////////////////////////////////

    target_dir = "I:\\GPT-Talker\\datasets\\processed\\DailyTalk\\"
    lang = "EN"   # ZH
    root_dir = "I:\\GPT-Talker\\datasets\\raw\\DailyTalk\\"
    data_name = "DailyTalk"    # Dialogue Turn = 2

    # ////////////////////////////////////////////
    
    # Generate a .list file (slicer_opt.list)
    # Data format: [address of voice|speaker identity|language|corresponding text]

    txt_files = glob.glob(os.path.join(target_dir, '*.txt'))  # .lab
    if not txt_files:
        txt_files = glob.glob(os.path.join(target_dir, '*.lab'))

    folders = get_all_folders(root_dir)
    data = {}
    content_list = []   

    for folder in folders:
        print(folder)
        wav_files = get_all_wav_in_dir(folder)

        for wav in wav_files:
            print("------new dialogue-------")
            print(wav)
            wav_basename = os.path.basename(wav)
            index = wav_basename.split("_")[0]     
            speaker = wav_basename.split("_")[1]  
            dialogue = wav_basename.split("_")[2].strip("d").strip(".wav") 
        
            txt_path = os.path.join(folder,wav_basename.replace(".wav",".txt"))
            if not txt_path:
                txt_path = glob.glob(os.path.join(target_dir, '*.lab'))

            with open(txt_path, "r", encoding="utf8") as file:
                content = file.read()

            wav_path = wav 
            speaker = data_name+"_"+lang+"_"+speaker 
            language = lang 
            text = content.strip()  

            content_list.append(wav_path+"|"+speaker+"|"+language+"|"+text)

    # Calling the function to write to a .list file
    write_to_list_file(os.path.join(target_dir,"slicer_opt.list"), content_list)