# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/model/t2s_model.py
import torch
from tqdm import tqdm
import numpy as np
from AR.models.utils import make_pad_mask
from AR.models.utils import topk_sampling,sample,logits_to_probs,multinomial_sample_one_no_sync
from AR.modules.embedding import SinePositionalEmbedding
from AR.modules.embedding import TokenEmbedding
from AR.modules.transformer import LayerNorm
from AR.modules.transformer import TransformerEncoder
from AR.modules.transformer import TransformerEncoderLayer
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy
import pprint
import gc

default_config = {
    "embedding_dim": 512,
    "hidden_dim": 512,
    "num_head": 8,
    "num_layers": 12,
    "num_codebook": 8,
    "p_dropout": 0.0,
    "vocab_size": 1024 + 1,
    "phoneme_vocab_size": 512,
    "EOS": 1024
}


class Text2SemanticDecoder(nn.Module):
    def __init__(self, config, norm_first=False, top_k=3):
        super(Text2SemanticDecoder, self).__init__()
        self.model_dim = config['model']["hidden_dim"]
        self.embedding_dim = config['model']["embedding_dim"]
        self.num_head = config['model']["head"]
        self.num_layers = config['model']["n_layer"]
        self.norm_first = norm_first
        self.vocab_size = config['model']["vocab_size"]
        self.phoneme_vocab_size = config['model']["phoneme_vocab_size"]
        self.p_dropout = config['model']["dropout"]
        self.EOS = config['model']["EOS"]
        self.norm_first = norm_first
        assert self.EOS == self.vocab_size - 1
        # should be same as num of kmeans bin
        # assert self.EOS == 1024
        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(
            self.embedding_dim, self.phoneme_vocab_size, self.p_dropout)
        self.ar_text_position = SinePositionalEmbedding(
            self.embedding_dim, dropout=0.1, scale=False, alpha=True)
        self.ar_audio_embedding = TokenEmbedding(
            self.embedding_dim, self.vocab_size, self.p_dropout)
        self.ar_audio_position = SinePositionalEmbedding(
            self.embedding_dim, dropout=0.1, scale=False, alpha=True)

        self.h = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=self.num_head,
                dim_feedforward=self.model_dim * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first, ),
            num_layers=self.num_layers,
            norm=LayerNorm(self.model_dim) if norm_first else None, )

        self.ar_predict_layer = nn.Linear(
            self.model_dim, self.vocab_size, bias=False)
        self.loss_fct = nn.CrossEntropyLoss(reduction='sum')

        self.ar_accuracy_metric = MulticlassAccuracy(
            self.vocab_size,
            top_k=top_k,
            average="micro",
            multidim_average="global",
            ignore_index=self.EOS,)
        
 
    def forward(self, x, x_lens, y, y_lens, bert_feature, phone_len, bert_feature_len, speaker_list, semantic_len):

        max_speakers_len = max(len(sublist) for sublist in speaker_list)

        # Splicing conversation history information
        s_x_lens = [] 
        s_y_lens = []
        sx_sum_lens = []  # the total length of the text of the conversations
        sy_sum_lens = []  # the total length of the speech of the conversations
        sxsy_sum_lens = []

        for index in range(0, len(x)):
            s_x_len = []
            s_y_len = []

            sx_sum_len = 0
            sy_sum_len = 0
            for s in range(0, max_speakers_len):
                if(s < len(phone_len[index])):
                    s_x_len.append(phone_len[index][s])  
                    sx_sum_len += phone_len[index][s]

                if(s == (len(phone_len[index])-1)):
                    s_y_len.append(semantic_len[index][s])
                    sy_sum_len += semantic_len[index][s]

            sx_sum_lens.append(sx_sum_len)
            sy_sum_lens.append(sy_sum_len)
            sxsy_sum_lens.append(sx_sum_len + sy_sum_len)
            
            s_x_lens.append(s_x_len)
            s_y_lens.append(s_y_len)
        
        max_dialog_len = max([x + y for x, y in zip(sx_sum_lens, sy_sum_lens)])
        s_x_s_y_embeds = None

        for index in range(0,len(x)): 
            start_pos_x = 0
            start_pos_y = sum(s_y_lens[index][:-1])
            s_x_s_y_embed = None 

            for s in range(0, max_speakers_len):
                if(s < len(semantic_len[index])):
                    x_id = x[index][start_pos_x:start_pos_x + phone_len[index][s]].unsqueeze(0)
                    x_bert_feature = bert_feature.transpose(1,2)[index][start_pos_x:start_pos_x + phone_len[index][s]].unsqueeze(0)
                    start_pos_x = start_pos_x + phone_len[index][s]

                    s_x_embed = self.ar_text_embedding(x_id)   # [B,T,512]
                    s_x_embed = s_x_embed + self.bert_proj(x_bert_feature)
                    s_x_embed = self.ar_text_position(s_x_embed)

                    y_id = None
                    # ///////////// speech part ////////////////
                    if(s == (len(semantic_len[index])-1)):
                        y_id = y[index][start_pos_y:start_pos_y + semantic_len[index][s]]
                        cut_y_len = max_dialog_len - (sx_sum_lens[index] + sy_sum_lens[index])
                        if(cut_y_len > 0):
                            zeros_1024 = torch.tensor([1024] * cut_y_len).to(y.device)
                            y_id = torch.cat((y_id, zeros_1024), dim=0).unsqueeze(0)
                        else:
                            y_id = y_id.unsqueeze(0)

                    if s < (len(semantic_len[index])-1):
                        if(s_x_s_y_embed == None):
                            s_x_s_y_embed = s_x_embed
                        else:
                            s_x_s_y_embed = torch.cat((s_x_s_y_embed, s_x_embed), dim=1)

                    else:
                        s_y_embed = self.ar_audio_embedding(y_id)
                        s_y_embed = self.ar_audio_position(s_y_embed) 
                        if(s_x_s_y_embed == None):
                            s_x_s_y_embed = torch.cat((s_x_embed, s_y_embed), dim=1)
                        else:
                            s_x_s_y_embed = torch.cat((s_x_s_y_embed, s_x_embed, s_y_embed), dim=1)

            if s_x_s_y_embeds == None:
                s_x_s_y_embeds = s_x_s_y_embed
            else:
                s_x_s_y_embeds = torch.cat((s_x_s_y_embeds, s_x_s_y_embed), dim=0)

        sy_sum_lens = torch.tensor(y_lens).to(y.device)
        dialog_mask = make_pad_mask(y_lens)
        dialog_mask_int = dialog_mask.type(torch.int64)
        dialog_codes = y.type(torch.int64) * (1 - dialog_mask_int)
   
        # AR Decoder
        dialog_y, dialog_targets = self.pad_y_eos(dialog_codes, dialog_mask_int, eos_id=self.EOS)

        x_attn_mask = F.pad(
            torch.zeros((1, 1), dtype=torch.bool, device=x.device),
            (0, max_dialog_len-1),
            value=True,
        )
        y_attn_mask = F.pad(
            torch.triu(
                torch.ones(max_dialog_len-1, max_dialog_len-1, dtype=torch.bool, device=x.device),
                diagonal=1,
            ),
            (1, 0),
            value=False,
        )
        sxsy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0).to(x.device)
        sxsy_sum_lens = torch.tensor(sxsy_sum_lens).to(y.device)
        xy_padding_mask = make_pad_mask(sxsy_sum_lens)
        ar_xy_padding_mask = xy_padding_mask
        # print(ar_xy_padding_mask)

        xy_len = sxsy_sum_lens.max()
        bsz, src_len = x.shape[0], xy_len

        _xy_padding_mask = (
            ar_xy_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, self.num_head, -1, -1)
            .reshape(bsz * self.num_head, 1, src_len)
        )

        sxsy_attn_mask = sxsy_attn_mask.logical_or(_xy_padding_mask)
        new_attn_mask = torch.zeros_like(sxsy_attn_mask, dtype=s_x_s_y_embeds.dtype)

        new_attn_mask.masked_fill_(sxsy_attn_mask, float("-inf"))
        sxsy_attn_mask = new_attn_mask

        xy_pos = s_x_s_y_embeds
        xy_dec, _ = self.h(
            (xy_pos, None),
            mask = sxsy_attn_mask,
        )

        y_dec = None   
        target_dec = None 
    
        for bsz_index in range(0,len(x)):
            y_dec_dia = None 
            target_dec_dia = None
            start_dec_pos = 0
            start_tar_pos = 0

            start_dec_pos = sum(s_x_lens[bsz_index][:])   
            end_dec_pos = start_dec_pos + s_y_lens[bsz_index][-1]  

            start_tar_pos = sum(s_y_lens[bsz_index][:-1])   
            end_tar_pos = start_tar_pos + s_y_lens[bsz_index][-1] 
            
            y_dec_dia = xy_dec[bsz_index][start_dec_pos:end_dec_pos,:]
            target_dec_dia = dialog_targets[bsz_index][start_tar_pos:end_tar_pos]


            if(sy_sum_lens.max() > y_dec_dia.shape[0]):
                size = (sy_sum_lens.max() - y_dec_dia.shape[0], 512)
                value = 1024
                pad_dec_y = torch.full(size, value).to(y.device)
                y_dec_dia = torch.cat([y_dec_dia, pad_dec_y], dim=0)

                cut = sy_sum_lens.max() - target_dec_dia.shape[0]
                cut = cut.item()
                pad_tar_y = torch.full((cut,), 1024).to(y.device)

                target_dec_dia = torch.cat([target_dec_dia, pad_tar_y], dim=0)


            if(y_dec == None):
                y_dec = y_dec_dia.unsqueeze(0)
                target_dec = target_dec_dia.unsqueeze(0)
            else:
                y_dec = torch.cat([y_dec, y_dec_dia.unsqueeze(0)], dim=0)
                target_dec = torch.cat([target_dec, target_dec_dia.unsqueeze(0)], dim=0)

        logits = self.ar_predict_layer(y_dec).permute(0, 2, 1)
        loss = F.cross_entropy(logits, target_dec, reduction="sum")
        acc = self.ar_accuracy_metric(logits.detach(), target_dec).item()

        return loss, acc

    def infer_new(self,
            history_phone_id,
            history_len,
            history_bert,
            history_semantic,
            current_phones,
            current_bert,
            current_len,
            top_k: int=-100,
            early_stop_num: int=-1,
            temperature: float=1.0):
        
        h_x_embed = []
        h_index = 0
        for h_phone in history_phone_id:
            x1 = self.ar_text_embedding(h_phone)
            x1 = x1 + self.bert_proj(history_bert[h_index].transpose(1,2))
            x1 = self.ar_text_position(x1)
            h_index += 1
            h_x_embed.append(x1)

        
        x2 = self.ar_text_embedding(current_phones)
        x2 = x2 + self.bert_proj(current_bert.transpose(1,2))
        x2 = self.ar_text_position(x2)

        stop = False
        history_len_all = 0
        for h_txt_len in history_len:
            history_len_all += h_txt_len
        
       
        for h_semantic in history_semantic:
            history_len_all += h_semantic.shape[1]
        
        stop = False
        idx = 0
        prompts = y = history_semantic[-1]  
        prefix_len = history_semantic[-1].shape[1]

        
        history_semantic_embed = []
        for h_semantic in history_semantic:
            h_y_h_embed = self.ar_audio_embedding(h_semantic)
            h_y_pos = self.ar_audio_position(h_y_h_embed)
            history_semantic_embed.append(h_y_pos)
            
        for _ in tqdm(range(800)):
            
            x_attn_mask = torch.zeros((1, 1), dtype=torch.bool)
            x_attn_mask_pad = F.pad(
                x_attn_mask,
                (0, idx + history_len_all + current_len -1),
                value=True, )
            y_attn_mask = F.pad(
                torch.triu(
                    torch.ones(idx+history_len_all+current_len-1, idx+history_len_all+current_len-1, dtype=torch.bool), diagonal=1),
                (1, 0),
                value=False, )
            xy_attn_mask = torch.concat(
                [x_attn_mask_pad, y_attn_mask], dim=0).to(y.device)
            
            
            if idx == 0:
                count = 0
                history = []
                for x in h_x_embed:
                    history.append(x)
                    y_pos = history_semantic_embed[count]
                    history.append(y_pos)
                    count+=1

                history.append(x2)

                xy_pos = torch.concat(history, dim=1)
            else:
                y_emb = self.ar_audio_embedding(y)
                y_pos_current = self.ar_audio_position(y_emb[:,-idx:])

                count = 0
                history = []
                for x in h_x_embed:
                    history.append(x)
                    y_pos = history_semantic_embed[count]
                    history.append(y_pos)
                    count+=1

                history.append(x2)
                history.append(y_pos_current)
                xy_pos = torch.concat(history, dim=1)
                

            idx += 1
            xy_dec, _ = self.h(
                (xy_pos, None),
                mask=xy_attn_mask,)
            logits = self.ar_predict_layer(xy_dec[:, -1])
            samples = topk_sampling(
                logits, top_k=top_k, top_p=1.0, temperature=temperature)

            if early_stop_num != -1 and (y.shape[1] - prefix_len
                                         ) > early_stop_num:
                print("use early stop num:", early_stop_num)
                stop = True

            if torch.argmax(
                    logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                stop = True

            if stop:
                if prompts.shape[1] == y.shape[1]:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                    print('bad zero prediction')
                print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                break
            y = torch.concat([y, samples], dim=1)

        return y, idx
    

    def pad_y_eos(self, y, y_mask_int, eos_id):
        targets = F.pad(
            y, (0, 1), value=0) + eos_id * F.pad(
                y_mask_int, (0, 1), value=1)

        return targets[:, :-1], targets[:, 1:]