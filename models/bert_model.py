import torch
import os
from torch import nn
import torch.nn.functional as F
from torchcrf import CRF
from .modeling_bert import BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import CLIPVisionModel,ViTMAEModel

class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")#("facebook/vit-mae-base")#("openai/clip-vit-base-patch32")
    
    def forward(self,imgs,aux_imgs=True):
        prompt_guids = self.get_vision_prompt(imgs) #13x(bsz,50,768)

        if aux_imgs is not None: #aux_imgs: bszx3(num)x3x224x224
            aux_prompt_guids = []
            aux_imgs = aux_imgs.permute([1,0,2,3,4]) #3(num)xbszx3x224x224
            for i in range(len(aux_imgs)):
                aux_prompt_guid = self.get_vision_prompt(aux_imgs[i]) #13x(bsz,50,768)
                aux_prompt_guids.append(aux_prompt_guid)
            return prompt_guids, aux_prompt_guids #prompt_guids:(13,bsz,50,768) aux_prompt_guids:3(num)x13x(bsz,50,768)
        return prompt_guids, None

    def get_vision_prompt(self,x):
        prompt_guids = list(self.clip(x, output_hidden_states=True).hidden_states)
        return prompt_guids # 13x(bsz,seq_len,hidden_size)


class AMNetREModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(AMNetREModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_name)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.args = args

        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.bert.config.hidden_size*2, num_labels)
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.tokenizer = tokenizer
        
        self.init_params = []

        if self.args.use_prompt:
            self.image_model = ImageModel()

            self.encoder_conv =  nn.Sequential(
                                    nn.Linear(in_features= 41600, out_features=1000), 
                                    nn.Tanh(),
                                    nn.Linear(in_features=1000, out_features=12*2*768) #12*2*768
                                )

        if self.args.use_contrastive:
            self.temp = nn.Parameter(torch.ones([])*self.args.temp) #temp: 0.179
            self.temp_lamb = nn.Parameter(torch.ones([])*self.args.temp_lamb) #temp_lamb:0.5
            self.alpha = nn.Parameter(torch.ones([])*self.args.alpha) #0.88
            # self.vision_proj = nn.Linear(self.bert.config.hidden_size,self.args.embed_dim) #768→256
            # self.text_proj = nn.Linear(self.bert.config.hidden_size,self.args.embed_dim)
            
            self.vision_proj = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size,4*self.bert.config.hidden_size),
                nn.Tanh(),
                nn.Linear(4*self.bert.config.hidden_size,self.args.embed_dim)  
            )
            self.text_proj = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size,4*self.bert.config.hidden_size),
                nn.Tanh(),
                nn.Linear(4*self.bert.config.hidden_size,self.args.embed_dim)  
            )

        if self.args.use_matching:
            self.itm_head = nn.Sequential(
                nn.Linear(768,768*4),
                nn.Tanh(),
                nn.Linear(768*4,2)
            )
            
            
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        images=None,
        aux_imgs=None,
    ):

        bsz = input_ids.size(0)
        with torch.no_grad():
            images, aux_images = self.image_model(images, aux_imgs) #prompt_guids:13x(bsz,50,768) aux_prompt_guids:3(num)x13x(bsz,50,768)
        guids,image_guids, aux_image_guids = self.get_visual_prompt(images,aux_images) #prompt_guids：6x(key,value), image_guids:(13,bsz,50,768) aux_image_guids:(3,13,bsz,50,768)

        image_atts = torch.ones((bsz, guids[0][0].shape[2])).to(self.args.device) #(16,24)
        text_output = self.bert(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    mode = 'text',
        )    
        text_last_hidden_state = text_output.last_hidden_state.clone() #(16,768)
        fusion_output = self.bert(
                    inputs_embeds=text_last_hidden_state,
                    attention_mask=torch.cat((image_atts,attention_mask), dim=1),
                    past_key_values=guids,
                    output_attentions=True,
                    return_dict=True,
                    mode = 'fusion',
        )
        fusion_last_hidden_state,fusion_pooler_output = fusion_output.last_hidden_state, fusion_output.pooler_output #(16,768)
        bsz, _, hidden_size = fusion_last_hidden_state.shape
        entity_hidden_state = torch.Tensor(bsz, 2*hidden_size) # batch, 2*hidden
        for i in range(bsz):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            head_hidden = fusion_last_hidden_state[i, head_idx, :].squeeze()
            tail_hidden = fusion_last_hidden_state[i, tail_idx, :].squeeze()
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)
        entity_hidden_state = entity_hidden_state.to(self.args.device)
        logits = self.classifier(entity_hidden_state)
       
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            text_feats = self.text_proj(text_last_hidden_state[:,0,:]) #[CLS]
            image_feats = self.vision_proj(images[12][:,0,:]) #[CLS] (bsz,768)
            # text_feats = F.normalize(self.text_proj(text_last_hidden_state[:,0,:]))#[CLS]
            # image_feats = F.normalize(self.vision_proj(images[12][:,0,:])) #[CLS] (bsz,768) image_guids[12,:,0,:]
            cl_loss = self.get_contrastive_loss(text_feats,image_feats)
            #image_guids:(13,bsz,50,768) aux_image_guids:(3,13,bsz,50,768)
            neg,itm_lable = self.get_matching_loss(image_guids,aux_image_guids,image_atts,image_feats,text_last_hidden_state,attention_mask,text_feats) #(32,768)
            neg_output = self.itm_head(torch.cat([fusion_pooler_output, neg], dim=0)) #(3*bsz,2)
            matching_loss = loss_fn(neg_output,itm_lable)
            main_loss = loss_fn(logits, labels.view(-1))
            loss = 0.5*main_loss  + 0.1*matching_loss +0.4*cl_loss # 84.57
            return loss,logits
        return logits

    def get_visual_prompt(self, images=None, aux_images=None): #images:13x(bsz,50,768) aux_images:3(num)x13x(bsz,50,768)
        
        bsz = images[0].size(0)
        
        image_guids = torch.stack(images) #image_guids:(13,bsz,50,768)
        aux_image_guids = torch.stack([torch.stack(aux_image) for aux_image in aux_images]) #aux_image_guids:(3，13，bsz,50,768)

        prompt_guids = torch.cat(images, dim=1).view(bsz, self.args.prompt_len, -1) #(bsz,12, 41600)
        prompt_guids = self.encoder_conv(prompt_guids) #(bsz,12,12*2*768)
        split_prompt_guids = prompt_guids.split(768*2, dim=-1) #12x(bsz,12,768*2)

        aux_prompt_guids = [torch.cat(aux_image, dim=1).view(bsz, self.args.prompt_len, -1) for aux_image in aux_images]  # 3x(13,bsz, 12, 41600)
        aux_prompt_guids = [self.encoder_conv(aux_prompt_guid) for aux_prompt_guid in aux_prompt_guids] # 3x(bsz, 12, 12*2*768)
        split_aux_prompt_guids = [aux_prompt_guid.split(768*2, dim=-1) for aux_prompt_guid in aux_prompt_guids] #3x(12x(bsz,12,768*2))

        result = []
        for idx in range(6):  # 6
            aux_key_vals = []   # 3 x [bsz, 12, 768*2]
            for split_aux_prompt_guid in split_aux_prompt_guids:
                aux_key_vals.append(split_aux_prompt_guid[idx])
            key_val = [split_prompt_guids[idx+6]]  + aux_key_vals #4x(bsz,12,768*2))
            key_val = torch.cat(key_val, dim=1) #(bsz,12*4,768*2) 
            key_val = key_val.split(768, dim=-1) #2*(bsz,6*4,768)
            key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[1].reshape(bsz, 12, -1, 64).contiguous()  # bsz, 12, 4, 64
            temp_dict = (key, value)
            result.append(temp_dict)
        return result, image_guids, aux_image_guids #image_guids:(13,bsz,50,768) aux_image_guids:#(3，13，bsz,50,768)
        
    
    def get_matching_loss(self,images,aux_images,image_atts,image_feat,text_embeds,text_atts,text_feat):
        #images:(13,bsz,50,768) aux_images:(3,13,bsz,50,768) 
        bsz = images.size(1)
        with torch.no_grad():
            sim_i2t = image_feat @ text_feat.t() / self.temp 
            sim_t2i = text_feat @ image_feat.t() / self.temp
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-5
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-5
            weights_i2t.fill_diagonal_(0) #对角线元素置零
            weights_t2i.fill_diagonal_(0)
        image_embeds_neg = []
        image_atts_neg = []
        aux_image_embeds_neg = []
        image_neg_list = []
        aux_image_neg_list = []
        
        for b in range(bsz):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item() #采样并返回下标
            image_embeds_neg.append(images.permute(1,0,2,3)[neg_idx]) #得到图片负样本的embed (bsz)*(13,50,768)
            image_atts_neg.append(image_atts[neg_idx]) #将负样本的下标作为嵌入
            aux_image_embeds_neg.append(aux_images.permute(2,1,0,3,4)[neg_idx]) #bszx(3,13,seq_len,embed) 

        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)
        image_atts_neg = torch.stack(image_atts_neg,dim=0) #(bsz,24)
        aux_image_embeds_neg = torch.stack(aux_image_embeds_neg,dim=0) #(bsz,13,3,50,768)
        
        text_embeds_neg = []
        text_atts_neg = []

        for b in range(bsz):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()#采样并返回下标,得到负样本的索引neg_idx
            text_embeds_neg.append(text_embeds[neg_idx]) ##得到文本负样本的嵌入,(seq_len,embed) → (bsz)x(seq_len,embed)
            text_atts_neg.append(text_atts[neg_idx]) #将负样本mask (seq_len)→ (bsz)x(seq_len)

        text_embeds_neg = torch.stack(text_embeds_neg,dim=0) #(bsz,seq_len,768)
        text_atts_neg = torch.stack(text_atts_neg,dim=0) #(bsz,seq_len)
        
        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0) #(2*bsz,seq_len,768)
        text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0) #(2*bsz,seq_len)

        image_embeds_all = torch.cat([image_embeds_neg, images.permute(1,0,2,3)],dim=0) #(2*bsz,13,50,768)
        aux_image_embeds_all  =torch.cat([aux_image_embeds_neg, aux_images.permute(2,1,0,3,4)],dim=0) #(2*bsz,13,3,50,768)
        image_atts_all = torch.cat([image_atts_neg, image_atts], dim=0) #(2*bsz,seq_len,768)
        
        for aux_image_neg in aux_image_embeds_all.permute(2,0,1,3,4): # aux_images_neg:(bsz,13,50,768)
            aux_image_neg_list.append([aux_image for aux_image in aux_image_neg.permute(1,0,2,3)]) #3x(13x(2*bsz,50,768))

        for image_neg in image_embeds_all.permute(1,0,2,3): #(bsz,13,50,768)
            image_neg_list.append(image_neg) #13x(2*bsz,50,768)
        
        neg_guids,_, _ = self.get_visual_prompt(image_neg_list,aux_image_neg_list) #prompt_guids：6x(key,value)

        neg = self.bert(
                    inputs_embeds=text_embeds_all,
                    attention_mask=torch.cat((image_atts_all,text_atts_all), dim=1),
                    past_key_values = neg_guids,
                    output_attentions = True,
                    return_dict=True,
                    mode = 'fusion',
        ).last_hidden_state[:,0,:]#[cls]
        itm_labels = torch.cat([torch.ones(bsz, dtype=torch.long),torch.zeros(2*bsz, dtype=torch.long)], dim=0).to(self.args.device)
        return neg,itm_labels
    
    def get_contrastive_loss(self,image_feat, text_feat):
        logits = text_feat @ image_feat.t() / self.temp
        bsz = text_feat.shape[0]
        labels = torch.arange(bsz, device=self.args.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        loss = loss_i2t + loss_t2i 
        return loss        
class AMNetNERModel(nn.Module):
    def __init__(self, label_list, args):
        super(AMNetNERModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_name)
        self.args = args
        if self.args.use_prompt:
            self.image_model = ImageModel()
            self.encoder_conv =  nn.Sequential(
                                    nn.Linear(in_features= 41600, out_features=1000), 
                                    nn.Tanh(),
                                    nn.Linear(in_features=1000, out_features=12*2*768) #6*2*768
                                )
        if self.args.use_contrastive:
            self.temp = nn.Parameter(torch.ones([])*self.args.temp) #temp: 0.179
            self.temp_lamb = nn.Parameter(torch.ones([])*self.args.temp_lamb) #temp_lamb:0.5
            self.alpha = nn.Parameter(torch.ones([])*self.args.alpha) #0.88
            
            
            self.vision_proj = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size,4*self.bert.config.hidden_size),
                nn.Tanh(),
                nn.Linear(4*self.bert.config.hidden_size,self.args.embed_dim)  
            )
            self.text_proj = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size,4*self.bert.config.hidden_size),
                nn.Tanh(),
                nn.Linear(4*self.bert.config.hidden_size,self.args.embed_dim)  
            )
        self.num_labels  = len(label_list)  # pad
        self.crf = CRF(self.num_labels, batch_first=True)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.5)
        if self.args.use_matching:
            self.itm_head = nn.Sequential(
                nn.Linear(768,768*4),
                nn.Tanh(),
                nn.Linear(768*4,2)
            )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        images=None,
        aux_imgs=None,
    ):
        bsz = input_ids.size(0)
        with torch.no_grad():
            images, aux_images = self.image_model(images, aux_imgs) #prompt_guids:13x(bsz,50,768) aux_prompt_guids:3(num)x13x(bsz,50,768)
        guids,image_guids, aux_image_guids = self.get_visual_prompt(images,aux_images) #prompt_guids：6x(key,value), image_guids:(13,bsz,50,768) aux_image_guids:(3,13,bsz,50,768)
        image_atts = torch.ones((bsz, guids[0][0].shape[2])).to(self.args.device)
        text_output = self.bert(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    mode = 'text',
        )
        text_last_hidden_state = text_output.last_hidden_state.clone()
        fusion_output = self.bert(
                    inputs_embeds=text_last_hidden_state,
                    attention_mask=torch.cat((image_atts,attention_mask), dim=1),
                    past_key_values=guids,
                    output_attentions=True,
                    return_dict=True,
                    mode = 'fusion',
        )
        fusion_last_hidden_state,fusion_pooler_output = fusion_output.last_hidden_state, fusion_output.pooler_output #(16,768)
        fusion_last_hidden_state = self.dropout(fusion_last_hidden_state) # bsz, len, hidden
        emissions = self.fc(fusion_last_hidden_state)    # bsz, len, labels
        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            text_feats = self.text_proj(text_last_hidden_state[:,0,:]) #[CLS]
            image_feats = self.vision_proj(images[12][:,0,:]) #[CLS] (bsz,768)
            cl_loss = self.get_contrastive_loss(text_feats,image_feats)
            neg,itm_lable = self.get_matching_loss(image_guids,aux_image_guids,image_atts,image_feats,text_last_hidden_state,attention_mask,text_feats) #(32,768)
            output = self.itm_head(torch.cat([fusion_pooler_output, neg], dim=0)) #(3*bsz,2)
            matching_loss = loss_fn(output,itm_lable)
            main_loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            # loss = self.alpha*main_loss + (1-self.alpha)*(cl_loss + matching_loss) #87.85 twitter17
            loss = 0.9*main_loss+0.1*matching_loss+0.1*cl_loss #87.56 twitter17
            # loss = 0.9*main_loss+0.1*cl_loss #87.51
            # loss = 0.9*main_loss + 0.1*matching_loss #87.51
            #loss = 0.9*main_loss + 0.01*matching_loss +0.1*cl_loss#87.23
            # loss = 0.8*main_loss+0.1*cl_loss #87.32
            # loss = 0.9*main_loss+0.1*matching_loss+0.05*cl_loss #87.23
            # loss = 0.7*main_loss+0.2*matching_loss+0.1*cl_loss #86.93
            #loss = 0.9051*main_loss + 0.0949*cl_loss + 0.0949*matching_loss #87.29

            # loss = 0.9*main_loss+0.1*matching_loss+0.1*cl_loss #75.53
            # loss = 0.8*main_loss+0.1*matching_loss+0.1*cl_loss #86.96
            # loss = 0.7*main_loss + 0.4*cl_loss + 0.1*matching_loss
            # loss = 0.8*main_loss+ 0.1*cl_loss + 0.1*matching_loss #76.00 twitter15
            # loss=0.9*main_loss
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )
    def get_visual_prompt(self, images=None, aux_images=None): #images:13x(bsz,50,768) aux_images:3(num)x13x(bsz,50,768)
        
        bsz = images[0].size(0)
        
        image_guids = torch.stack(images) #image_guids:(13,bsz,50,768)
        aux_image_guids = torch.stack([torch.stack(aux_image) for aux_image in aux_images]) #aux_image_guids:(3，13，bsz,50,768)

        prompt_guids = torch.cat(images, dim=1).view(bsz, self.args.prompt_len, -1) #(bsz,12, 41600)
        prompt_guids = self.encoder_conv(prompt_guids) #(bsz,12,12*2*768)
        split_prompt_guids = prompt_guids.split(768*2, dim=-1) #12x(bsz,12,768*2)

        aux_prompt_guids = [torch.cat(aux_image, dim=1).view(bsz, self.args.prompt_len, -1) for aux_image in aux_images]  # 3x(13,bsz, 12, 41600)
        aux_prompt_guids = [self.encoder_conv(aux_prompt_guid) for aux_prompt_guid in aux_prompt_guids] # 3x(bsz, 12, 12*2*768)
        split_aux_prompt_guids = [aux_prompt_guid.split(768*2, dim=-1) for aux_prompt_guid in aux_prompt_guids] #3x(12x(bsz,12,768*2))

        result = []
        for idx in range(6):  # 6
            aux_key_vals = []   # 3 x [bsz, 12, 768*2]
            for split_aux_prompt_guid in split_aux_prompt_guids:
                aux_key_vals.append(split_aux_prompt_guid[idx+6])
            key_val = [split_prompt_guids[idx]]  + aux_key_vals #4x(bsz,12,768*2))
            key_val = torch.cat(key_val, dim=1) #(bsz,12*4,768*2) 
            key_val = key_val.split(768, dim=-1) #2*(bsz,6*4,768)
            key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[1].reshape(bsz, 12, -1, 64).contiguous()  # bsz, 12, 4, 64
            temp_dict = (key, value)
            result.append(temp_dict)
        return result, image_guids, aux_image_guids #image_guids:(13,bsz,50,768) aux_image_guids:#(3，13，bsz,50,768)
    
    def get_matching_loss(self,images,aux_images,image_atts,image_feat,text_embeds,text_atts,text_feat):
        #images:(13,bsz,50,768) aux_images:(3,13,bsz,50,768) 
        bsz = images.size(1)
        with torch.no_grad():
            sim_i2t = image_feat @ text_feat.t() / self.temp 
            sim_t2i = text_feat @ image_feat.t() / self.temp
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-5
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-5
            weights_i2t.fill_diagonal_(0) #对角线元素置零
            weights_t2i.fill_diagonal_(0)
        image_embeds_neg = []
        image_atts_neg = []
        aux_image_embeds_neg = []
        image_neg_list = []
        aux_image_neg_list = []
        
        for b in range(bsz):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item() #采样并返回下标
            # neg_idx = torch.argmax(weights_t2i[b]).item() #采样并返回下标
            image_embeds_neg.append(images.permute(1,0,2,3)[neg_idx]) #得到图片负样本的embed (bsz)*(13,50,768)
            image_atts_neg.append(image_atts[neg_idx]) #将负样本的下标作为嵌入
            aux_image_embeds_neg.append(aux_images.permute(2,1,0,3,4)[neg_idx]) #bszx(3,13,seq_len,embed) 

        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)
        image_atts_neg = torch.stack(image_atts_neg,dim=0) #(bsz,24)
        aux_image_embeds_neg = torch.stack(aux_image_embeds_neg,dim=0) #(bsz,13,3,50,768)
        
        text_embeds_neg = []
        text_atts_neg = []

        for b in range(bsz):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()#采样并返回下标,得到负样本的索引neg_idx
            # neg_idx = torch.argmax(weights_i2t[b]).item() #采样并返回下标
            text_embeds_neg.append(text_embeds[neg_idx]) ##得到文本负样本的嵌入,(seq_len,embed) → (bsz)x(seq_len,embed)
            text_atts_neg.append(text_atts[neg_idx]) #将负样本mask (seq_len)→ (bsz)x(seq_len)

        text_embeds_neg = torch.stack(text_embeds_neg,dim=0) #(bsz,seq_len,768)
        text_atts_neg = torch.stack(text_atts_neg,dim=0) #(bsz,seq_len)
        
        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0) #(2*bsz,seq_len,768)
        text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0) #(2*bsz,seq_len)

        image_embeds_all = torch.cat([image_embeds_neg, images.permute(1,0,2,3)],dim=0) #(2*bsz,13,50,768)
        aux_image_embeds_all  =torch.cat([aux_image_embeds_neg, aux_images.permute(2,1,0,3,4)],dim=0) #(2*bsz,13,3,50,768)
        image_atts_all = torch.cat([image_atts_neg, image_atts], dim=0) #(2*bsz,seq_len,768)
        
        for aux_image_neg in aux_image_embeds_all.permute(2,0,1,3,4): # aux_images_neg:(bsz,13,50,768)
            aux_image_neg_list.append([aux_image for aux_image in aux_image_neg.permute(1,0,2,3)]) #3x(13x(2*bsz,50,768))

        for image_neg in image_embeds_all.permute(1,0,2,3): #(bsz,13,50,768)
            image_neg_list.append(image_neg) #13x(2*bsz,50,768)
        
        neg_guids,_, _ = self.get_visual_prompt(image_neg_list,aux_image_neg_list) #prompt_guids：6x(key,value)

        neg = self.bert(
                    inputs_embeds=text_embeds_all,
                    attention_mask=torch.cat((image_atts_all,text_atts_all), dim=1),
                    past_key_values = neg_guids,
                    output_attentions = True,
                    return_dict=True,
                    mode = 'fusion',
        ).last_hidden_state[:,0,:]#[cls] (2xbsz,768)
        itm_labels = torch.cat([torch.ones(bsz, dtype=torch.long),torch.zeros(2*bsz, dtype=torch.long)], dim=0).to(self.args.device) # 3*bsz
        return neg,itm_labels
    
    def get_contrastive_loss(self,image_feat, text_feat):
        logits = text_feat @ image_feat.t() / self.temp
        bsz = text_feat.shape[0]
        labels = torch.arange(bsz, device=self.args.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        loss = loss_i2t + loss_t2i 
        return loss 