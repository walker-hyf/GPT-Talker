import os,sys,numpy as np
import traceback
from scipy.io import wavfile
from my_utils import load_audio
from slicer2 import Slicer

def slice(inp,opt_root,threshold,min_length,min_interval,hop_size,max_sil_kept,_max,alpha,i_part,all_part):
    os.makedirs(opt_root,exist_ok=True)
    if os.path.isfile(inp):
        input=[inp]
    elif os.path.isdir(inp):
        input=["%s/%s"%(inp,name)for name in sorted(list(os.listdir(inp)))]
    else:
        return "Input path exists but is neither a file nor a folder"
    slicer = Slicer(
        sr=32000,
        threshold=      int(threshold),  
        min_length=     int(min_length),  
        min_interval=   int(min_interval), 
        hop_size=       int(hop_size), 
        max_sil_kept=   int(max_sil_kept), 
    )
    _max=float(_max)
    alpha=float(alpha)
    for inp_path in input[int(i_part)::int(all_part)]:
        try:
            name = os.path.basename(inp_path)
            audio = load_audio(inp_path, 32000)
            for chunk, start, end in slicer.slice(audio):
                tmp_max = np.abs(chunk).max()
                if(tmp_max>1):chunk/=tmp_max
                chunk = (chunk / tmp_max * (_max * alpha)) + (1 - alpha) * chunk
                wavfile.write(
                    "%s/%s_%s_%s.wav" % (opt_root, name, start, end),
                    32000,
                    (chunk * 32767).astype(np.int16),
                )
        except:
            print(inp_path,"->fail->",traceback.format_exc())
    return "When execution is complete, check the output file"

print(slice(*sys.argv[1:]))