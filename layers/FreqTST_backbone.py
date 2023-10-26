# This file is licensed under the GNU General Public License (GPL) version 2.0.
# See the LICENSE file or https://www.gnu.org/licenses/gpl-2.0.html for more details.

__all__ = ['FreqTST_ci_tfi']

from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN
from models import DLinear

#from collections import OrderedDict
from layers.PatchTST_layers import *


#Super class of patched frequncy TST
class FreqTST_patched(nn.Module):
    def __init__(self, c_in:int,
                n_freq:int,
                context_window:int, target_window:int, 
                patch_len:int, stride:int,
                max_seq_len:Optional[int]=1024, 
                n_decomp:int=0, 
                n_concat_td:int=0,
                b_base_dec =True,
                b_learn_freq = True,
                padding_patch=None, **kwargs):
        #n_decomp: 0 for no decomposition, 1 for learnable decomposition, 2 for muti-decomposition proposed in MICN, 3 for Dlinear decomposition
        super().__init__()
        self.context_window=context_window
        self.target_window=target_window
        self.n_freq = n_freq
        self.c_in = c_in
        self.c_out = c_in
        self.b_base_dec = b_base_dec
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num =int((context_window-patch_len)/stride) +1
        self.padding_patch = padding_patch 
        self.n_decomp = n_decomp
        if padding_patch=='end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            self.patch_num += 1
        self.max_seq_len = max_seq_len
        if n_concat_td > self.patch_num:
            n_concat_td = self.patch_num
            print("Warning: n_concat_td > self.patch_num in FreqTST_patched, reset to {}".format(self.patch_num))  
        self.n_concat_td = n_concat_td      
        #Frequency
        if self.n_freq !=0:
            if not b_learn_freq:
                print("Frequency learning disabled")
            if n_freq <= self.patch_num-1:
                self.freqs = nn.Parameter(torch.tensor(np.linspace(0, n_freq-1, n_freq, dtype=np.float32)/self.patch_num).reshape((self.n_freq,1)), requires_grad=b_learn_freq)
            else:
                self.freqs = nn.Parameter(torch.tensor(np.linspace(0, n_freq-1, n_freq, dtype=np.float32)/n_freq).reshape((self.n_freq,1)), requires_grad=b_learn_freq)
            self.mul_freq_in=(torch.tensor(np.pi*(np.linspace(0, self.patch_num-1, self.patch_num)+0.5),dtype=torch.float32)\
                .reshape(1,self.patch_num)).repeat(n_freq,1)
            freqs_full=np.linspace(0,1,self.patch_num,endpoint=False,dtype=np.float32)
            freqs_full=freqs_full.reshape((self.patch_num,1))
            freqs_full=torch.tensor(freqs_full)
            mul_freq_full=(torch.tensor(np.pi*(np.linspace(0,self.patch_num-1,self.patch_num)+0.5),dtype=torch.float32)\
                        .reshape(1,self.patch_num)).repeat(self.patch_num,1)
            #base_full: [self.patch_num, self.patch_num]
            self.base_full=(2.0/self.patch_num)**0.5*torch.cos(mul_freq_full*freqs_full)
            self.base_full[0,:]=self.base_full[0,:] / 2**0.5
            self.trans_abs_sum=torch.zeros(self.patch_num, self.patch_len)
            #learnable base decay
            if self.b_base_dec:
                self.base_dec = nn.Parameter(torch.zeros(1), requires_grad=True)
        #Learnable decomposition
        if self.n_decomp ==1:
            self.decomp = Learnable_decomp(self.context_window, self.target_window, self.patch_len, l2_fac=0.001)
        elif self.n_decomp ==2:
            self.decomp = Multi_decomp(self.context_window, self.target_window)
        elif self.n_decomp ==3:
            self.decomp = Dlinear_decomp(self.context_window, self.target_window, self.c_in)
        # RevIn 
        self.revin_layer = RevIN(self.c_in, affine=False, subtract_last=False) 


    #Accumulate freqency domain amplitude
    def accum_freq_amp(self, z):           #z: [bs, context_window, c_in]
        if self.n_freq == 0:
            return
        z = self.revin_layer(z, 'norm')
        #z(out):  [bs, c_in, context_window]
        z = z.permute(0,2,1)
        #Padding before patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        #z(out):  [(bs*c_in), context_window]
        z = z.reshape([z.shape[0]*z.shape[1], z.shape[2]])
        #z(out):  [(bs*c_in), patch_num, patch_len]
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z_mean = torch.mean(z, dim=1, keepdim=True).detach()
        z = z-z_mean
        #tran_abs: [(bs*c_in), patch_num, patch_len]
        if self.base_full.device != z.device:
            self.base_full=self.base_full.to(z.device) 
        #tran_abs: [(bs*c_in), patch_num, patch_len]
        trans_abs = torch.abs(torch.matmul(self.base_full,z)) 
        #tran_abs(out): [patch_num, patch_len]
        trans_abs = trans_abs.mean(dim=0)
        if self.trans_abs_sum.device != z.device:
            self.trans_abs_sum=self.trans_abs_sum.to(z.device)  
        self.trans_abs_sum += trans_abs  
        
    #Compute initial freqencies
    def comp_ini_freq(self):
        if self.n_freq > self.patch_num or self.n_freq <=0:
            return
        trans_abs_mean = self.trans_abs_sum / self.trans_abs_sum.sum(dim=0)
        trans_abs_mean = trans_abs_mean.mean(dim=1)
        i_freqs=np.argsort(trans_abs_mean.detach().cpu().numpy())[-1:-self.n_freq-1:-1]
        if not 0 in i_freqs:    #TODO Ensure freq 0 is included
            i_freqs[-1] = 0
        self.i_freq_0 = i_freqs.tolist().index(0)
        print("i_freq_0",self.i_freq_0)
        freqs = (i_freqs/self.patch_num).astype(np.float32)
        freqs = torch.tensor(freqs).reshape((self.n_freq,1))
        with torch.no_grad():
            self.freqs[:,0] = freqs[:,0]
        self.show_freqs(self.n_freq)

    #Print current frequencies
    def show_freqs(self, n_disp=8):
        if self.n_freq <=0:
            return
        freqs_disp = self.freqs.detach().squeeze().cpu().numpy()*self.patch_num
        n_disp = min(n_disp, self.n_freq)
        if n_disp ==1:
            print("{:.2f}, ".format(freqs_disp))
            return
        print("\tFreqs: ", end="")
        for i_freq, freq in enumerate(freqs_disp):
            if i_freq >= n_disp:
                break
            print("{:.2f}, ".format(freq), end="")
        if n_disp < self.n_freq:
            print("...")
        else:
            print("")
        if self.b_base_dec:
            print("\tBase dec: ", self.base_dec.squeeze().item())



# Frequency domain transformer using channel independent (CI) model and time-freqency independent (TFI) model sequentially
# The marks are also used along with the data
class FreqTST_ci_tfi(FreqTST_patched):
    def __init__(self, c_in:int,
                n_freq:int,
                context_window:int, target_window:int, 
                patch_len:int, stride:int,
                d_compress_max:int=1, 
                c_mark: int=4, 
                use_mark: bool=False,
                sep_time_freq: bool=False,
                max_seq_len:Optional[int]=1024, 
                n_concat_td:int=0,
                mod_scal_tfi:float=1.0,
                n_layers:int=3,  n_layers_tfi:int=3,
                b_ori_router:bool=False,
                d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=False, store_attn:bool=False, pre_norm:bool=False, 
                fc_dropout:float=0., head_dropout = 0, 
                b_base_dec =False, padding_patch = None, 
                verbose:bool=False, **kwargs):
        #n_freq: number of frequency components
        #c_mark: number of channels of the marks
        #out channels has to be equal to in channels
        #d_compress_max: max width in the compressed (time) dimension for linear transformer
        #b_ori_router: enable original router mechanism the same as crossformer

        super().__init__(c_in,n_freq, context_window, target_window, patch_len, stride, max_seq_len=max_seq_len, 
                n_concat_td=n_concat_td, b_base_dec=b_base_dec, padding_patch=padding_patch, **kwargs)
        self.pre_norm = pre_norm
        self.d_model = d_model
        self.n_layers_tfi = n_layers_tfi
        self.c_mark = c_mark
        self.sep_time_freq = sep_time_freq
        ##Input encoding, bias is disabled to keep the mean to 0
        self.input_embedding_ci = nn.Linear(patch_len, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        ##Position embedding
        self.pos_embd_ci = nn.Parameter(torch.zeros((1,n_freq + n_concat_td,d_model)), requires_grad=True)
        #self.pos_embd_tfi = nn.Parameter(torch.zeros((1,self.c_in,d_model)), requires_grad=True)
        ##Backbone 
        self.encoder_ci = TSTEncoder(n_freq + n_concat_td, self.d_model, n_heads,
                                  d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout, pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)
        if self.c_in<=self.d_model and self.c_in <= d_compress_max:
            d_compress = None
        else:
            d_compress = min(self.d_model, d_compress_max)
        print('d_compress in mapped transformer', d_compress)
        assert mod_scal_tfi != 0, "Error: mod_scal_tfi=0 in FreqTST_ci_tfi"
        if mod_scal_tfi <0:
            d_ff_tfi = 0
            mod_scal_tfi = -mod_scal_tfi
        else:
            d_ff_tfi = int(d_ff * mod_scal_tfi)
        n_heads_tfi = max(int(n_heads * mod_scal_tfi),1)
        d_kv_tfi = max(int(d_model/n_heads * mod_scal_tfi), 1)
        if n_layers_tfi != 0 and not b_ori_router:
            #Use LRA layers
            self.encoder_tfi = TSTEncoder(self.c_in, self.d_model, n_heads_tfi, d_compress=d_compress, layer_type=5,
                    d_k=d_kv_tfi, d_v=d_kv_tfi, d_ff=d_ff_tfi, norm=norm, attn_dropout=attn_dropout, 
                    dropout=dropout, pre_norm=pre_norm, activation=act, res_attention=res_attention, 
                    n_layers=n_layers_tfi, store_attn=store_attn)   #n_tf = self.n_freq+self.n_concat_td
                    #Donot set n_tf to disable time-freq scaling
        elif n_layers_tfi != 0:
            #Use Router mechanism the same as Crossformer
            self.encoder_tfi = TSTEncoder(self.c_in, self.d_model, n_heads_tfi, d_compress=d_compress, layer_type=6,
                                    d_k=d_kv_tfi, d_v=d_kv_tfi, d_ff=d_ff_tfi, norm=norm, attn_dropout=attn_dropout, dropout=dropout, pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers_tfi, store_attn=store_attn)
            
            
        ##Head
        if not self.sep_time_freq:
            self.head_pred = nn.Sequential(
                nn.Flatten(start_dim=-2,end_dim=-1),
                get_activation_fn(act),
                nn.Dropout(head_dropout),
                nn.Linear((n_freq + n_concat_td)* d_model, target_window)
                )
        else:
            assert not use_mark, "Error: use_mark and sep_time_freq cannot be true at once"
            self.head_pred = nn.Sequential(
                nn.Flatten(start_dim=-2,end_dim=-1),
                get_activation_fn(act),
                nn.Dropout(head_dropout),
                nn.Linear(n_concat_td* d_model, target_window)
                )
            self.head_fd_2_td = nn.Linear(n_freq, n_concat_td)
            self.attn_mask = torch.zeros([n_freq+n_concat_td,n_freq+n_concat_td],dtype=torch.bool)
            self.attn_mask[n_freq:, :n_freq] = True
            self.attn_mask[:n_freq, n_freq:] = True
        ##Marks
        self.use_mark = use_mark
        if use_mark:
            #Position embedding of marks is added to the original data to save parameters
            self.pos_embd_mark = nn.Parameter(torch.zeros((1, context_window+target_window, self.c_mark)), requires_grad=True)
            self.mark_embedding=nn.Linear(c_mark, d_model, bias=False)
            self.encoder_mark = TSTEncoder(self.c_in, self.d_model, n_heads_tfi, d_compress=1, layer_type=2,
                                        d_k=d_kv_tfi, d_v=d_kv_tfi, d_ff=0, norm=norm, attn_dropout=attn_dropout, dropout=dropout, pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=1, store_attn=store_attn)
        
        



    #z: [bs, context_window, c_in], z_mark: [bs, context_window, c_mark], target_mark: [bs, target_window+n_label, c_mark]
    def forward(self, z, z_mark, target_mark, **kwargs):                  
        ##Decomposition
        if self.n_decomp != 0:
            z, trend_pred_fine = self.decomp(z)
        ##Norm
        z = self.revin_layer(z, 'norm')
        bs = z.shape[0]
        z = z.permute(0,2,1)                #z(out): [bs, c_in, context_window]
        ##Padding before patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        ##Patching
        #z(out):  [bs, c_in, patch_num, patch_len]
        z_patched = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z_patched = z_patched.permute(0,1,3,2)     #z_patched: [bs, c_in, patch_len, patch_num]    
        ##Customized DCT
        if self.n_freq != 0:
            if self.mul_freq_in.device != z.device:
                self.mul_freq_in = self.mul_freq_in.to(z.device)
            phase_shifts = self.mul_freq_in*self.freqs
            mul_cdct = torch.cos(phase_shifts)   #mul_cdct: [n_freq, patch_num]
            ##Base decay
            if self.b_base_dec: 
                mul_base_dec = torch.exp(self.base_dec*(phase_shifts-phase_shifts[:,-1:]))
                mul_cdct = mul_base_dec * mul_cdct
            mul_cdct = (2.0/self.patch_num)**0.5 * mul_cdct
            z_trans = torch.matmul(z_patched, mul_cdct.T)       #z_trans: [bs, c_in, patch_len, n_freq]
        ##Concat td data
        if self.n_concat_td != 0:
            z_concat_td = z_patched[:, :, :, -self.n_concat_td:]  #z_concat_td: [bs, c_in, patch_len, n_concat_td]
            if self.n_freq != 0:
                z_trans = torch.concat([z_trans, z_concat_td],dim=3)  #z_trans: [bs, c_in, patch_len, n_freq+n_concat_td]
            else:
                z_trans = z_concat_td
        ##CI embedding
        z_trans_ci = z_trans.reshape(bs*self.c_in, self.patch_len, self.n_freq+self.n_concat_td)
        z_trans_ci = z_trans_ci.permute(0,2,1) #z_trans: [bs*c_in, n_freq+n_concat_td, patch_len]
        z_enc_in_ci = self.input_embedding_ci(z_trans_ci)     #z_enc_in_ci: [bs*c_in, n_freq+n_concat_td, d_model]
        z_enc_in_ci = z_enc_in_ci + self.pos_embd_ci         #Positional embedding
        z_enc_in_ci = self.dropout(z_enc_in_ci) 
        ##CI backbone
        if not self.sep_time_freq:
            z_enc_ci = self.encoder_ci(z_enc_in_ci)              #z_enc_ci: [bs*c_in, n_freq+n_concat_td, d_model]
        else:
            if self.attn_mask.device != z.device:
                self.attn_mask = self.attn_mask.to(z.device)
            z_enc_ci = self.encoder_ci(z_enc_in_ci, attn_mask=self.attn_mask)
                
        ##TFI embedding and backbone
        z_enc_ci = z_enc_ci.reshape(bs, self.c_in, self.n_freq+self.n_concat_td, self.d_model)
        if self.n_layers_tfi != 0:
            #z_enc_in_tfi(out): [bs, n_freq+n_concat_td, c_in, d_model]
            z_enc_in_tfi = z_enc_ci.permute(0,2,1,3)
            #z_enc_tfi: [bs, n_freq+n_concat_td, c_in, d_model]
            #z_enc_in_tfi = z_enc_in_tfi+self.pos_embd_tfi           #Positional embedding
            #TFI backbone
            z_enc_tfi = self.encoder_tfi(self.dropout(z_enc_in_tfi))          #z_enc_tfi: [bs, n_freq+n_concat_td, c_in, d_model]
        else:
            z_enc_tfi = z_enc_ci.permute(0,2,1,3)
        if self.use_mark:
        ##Correcting prediction with marks
        #Truncate target_mark
            target_mark = target_mark[:, -self.target_window:, :]
            full_mark = torch.concat([z_mark, target_mark], dim=1)
            #mark_embed: [bs, context_window+target_window, c_mark]
            mark_embed = self.mark_embedding(full_mark + self.pos_embd_mark)
            #z_enc_mark: [bs, n_freq+n_concat_td, c_in, d_model]
            z_enc_mark = self.encoder_mark(z_enc_tfi, src_cross_in=mark_embed[:, :self.context_window, :], src_cross_out=mark_embed[:, -self.target_window:, :] )
            ###Linear prediction using head
            #z_enc_in_tfi_trans(out): [bs, c_in, n_freq+n_concat_td, d_model]
            z_enc_mark_trans = z_enc_mark.permute(0,2,1,3) 
            #z_pred: [bs, c_in, target_window]
            z_pred = self.head_pred(z_enc_mark_trans)
        else:
            if not self.sep_time_freq:
                z_pred = self.head_pred(z_enc_tfi.permute(0,2,1,3))
            else:
                z_enc_tfi = z_enc_tfi.permute(0,2,3,1) #z_enc_tfi: [bs, c_in, d_model, n_freq+n_concat_td]
                z_enc_fd = z_enc_tfi[:, :, :, : self.n_freq]
                z_enc_td = z_enc_tfi[:, :, :, self.n_freq :] + self.head_fd_2_td(z_enc_fd)
                z_pred = self.head_pred(z_enc_td)
                
        z_pred = z_pred.transpose(1,2)
        ##Denorm: z_pred: [bs, target_window, c_in]
        z_pred = self.revin_layer(z_pred, 'denorm')
        ##Add trend
        if self.n_decomp != 0:
            z_pred = z_pred + trend_pred_fine
        return z_pred
    
        

class Learnable_decomp(nn.Module):
    """
    Learnable decomposition block
    """

    def __init__(self, seq_len, pred_len, patch_len, l2_fac=0.001): #fc_dropout=0.0:
        super(Learnable_decomp, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.l2_fac = l2_fac
        assert np.mod(seq_len,patch_len)==0, "Error: mod(seq_len,patch_len)!=0 in Learnable_decomp"
        assert np.mod(pred_len,patch_len)==0, "Error: mod(pred_len,patch_len)!=0 in Learnable_decomp"
        self.patch_num = int(self.seq_len/self.patch_len)
        self.kernel_num = max(int(np.floor(np.log(seq_len-1)/np.log(2)))-1, 1)
        self.avg_pool_patch = nn.AvgPool1d(patch_len, stride=patch_len, padding=0)
        self.patch_num_pred = int(self.pred_len/self.patch_len)
        #self.fc_trend =nn.Sequential(nn.Dropout(fc_dropout), nn.Linear(self.patch_num, self.patch_num_pred, bias=False))
        self.fc_trend = nn.Linear(self.patch_num, self.patch_num_pred, bias=False)
        self.fc_trend.weight = nn.Parameter(torch.zeros([self.patch_num_pred, self.patch_num]), requires_grad=True) #(1/self.patch_num_pred) * torch.ones([self.patch_num_pred, self.patch_num])
        #Weights of modes before softmax
        self.importance_mods = nn.Parameter(torch.zeros(self.kernel_num, dtype=torch.float32), requires_grad=True)
        #self.weight_mods = nn.Parameter(torch.ones(self.kernel_num-1, dtype=torch.float32)/self.kernel_num, requires_grad=True) #no weight for mod 0 for the sum of weights is 1
        self.fn_pads = [nn.ReplicationPad1d(2**(i-1)) for i in range(1, self.kernel_num)] #no padding for mod 0
        self.avg_pools = [nn.AvgPool1d(2**i +1, stride=1, padding=0) for i in range(1,self.kernel_num)] #no avg_pool for mod 0
        

    def forward(self, x):
        #x: [bs, seq_len, c_in]
        x_transpose = x.permute(0,2,1) #x_transpose: [bs, c_in, seq_len]
        t_patched = self.avg_pool_patch(x_transpose) #t_patched: [bs, c_in, patch_num]
        weight_mods = F.softmax(self.importance_mods, dim=0)
        #trend: [bs, c_in, patch_num]
        trend = t_patched * weight_mods[0] #(1.0 - self.weight_mods.sum())
        for i_mod in range(0, self.kernel_num-1):
            t_padded = self.fn_pads[i_mod](t_patched)
            mod_cur = self.avg_pools[i_mod](t_padded)
            trend = trend + mod_cur * weight_mods[i_mod+1] #self.weight_mods[i_mod]
        trend_fine = F.interpolate(trend, scale_factor=self.patch_len, mode ='linear',align_corners=False)
        season = x_transpose - trend_fine #season: [bs, c_in, seq_len]
        season = season.permute(0,2,1) #season: [bs, seq_len, c_in]
        trend_pred = self.fc_trend(trend) #trend_pred: [bs, c_in, patch_num_pred]
        #trend_pred_fine: [bs, channel, pred_len]
        trend_pred_fine = F.interpolate(trend_pred, scale_factor=self.patch_len, mode ='linear',align_corners=False)
        trend_pred_fine = trend_pred_fine.permute(0,2,1)
        #L2 regularization for trend prediction
        if self.fc_trend.weight.grad != None:
            self.fc_trend.weight.grad.data.add_(self.l2_fac * self.fc_trend.weight.data)
        return season, trend_pred_fine #season: [bs, seq_len, c_in], trend_pred_fine: [bs, pred_len, channel]
    
    


class Multi_decomp(nn.Module):
    """
    multi-scale hybrid decomposition proposed by MICN
    """

    def __init__(self, seq_len, pred_len, kernel_sizes=[17,49], patch_len=1, l2_fac=0.001): #fc_dropout=0.0:
        super(Multi_decomp, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.l2_fac = l2_fac
        assert np.mod(seq_len,patch_len)==0, "Error: mod(seq_len,patch_len)!=0 in Multi_decomp"
        assert np.mod(pred_len,patch_len)==0, "Error: mod(pred_len,patch_len)!=0 in Multi_decomp"
        for kernel_size in kernel_sizes:
            assert kernel_size > 0 and (kernel_size -1) % 2==0, "Error: kernel_size in Multi_decomp should be positive odd"
        self.patch_num = int(self.seq_len/self.patch_len)
        self.kernel_num = len(kernel_sizes)
        self.avg_pool_patch = nn.AvgPool1d(patch_len, stride=patch_len, padding=0)
        self.patch_num_pred = int(self.pred_len/self.patch_len)
        #self.fc_trend =nn.Sequential(nn.Dropout(fc_dropout), nn.Linear(self.patch_num, self.patch_num_pred, bias=False))
        self.fc_trend = nn.Linear(self.patch_num, self.patch_num_pred, bias=False)
        self.fc_trend.weight = nn.Parameter((1/self.patch_num_pred) * torch.ones([self.patch_num_pred, self.patch_num]), requires_grad=True)
        self.fn_pads = [nn.ReplicationPad1d(int((kernel_size-1)/2)) for kernel_size in kernel_sizes]
        self.avg_pools = [nn.AvgPool1d(kernel_size, stride=1, padding=0) for kernel_size in kernel_sizes] #no avg_pool for mod 0
        

    def forward(self, x):
        #x: [bs, seq_len, c_in]
        x_transpose = x.permute(0,2,1) #x_transpose: [bs, c_in, seq_len]
        t_patched = self.avg_pool_patch(x_transpose) #t_patched: [bs, c_in, patch_num]
        #trend: [bs, c_in, patch_num]
        for i_mod in range(0, self.kernel_num):
            t_padded = self.fn_pads[i_mod](t_patched)
            mod_cur = self.avg_pools[i_mod](t_padded)
            if i_mod == 0:
                trend = mod_cur
            else:
                trend = trend + mod_cur
        trend = trend/self.kernel_num
        trend_fine = F.interpolate(trend, scale_factor=self.patch_len, mode ='linear',align_corners=False)
        season = x_transpose - trend_fine #season: [bs, c_in, seq_len]
        season = season.permute(0,2,1) #season: [bs, seq_len, c_in]
        trend_pred = self.fc_trend(trend) #trend_pred: [bs, c_in, patch_num_pred]
        #trend_pred_fine: [bs, channel, pred_len]
        trend_pred_fine = F.interpolate(trend_pred, scale_factor=self.patch_len, mode ='linear',align_corners=False)
        trend_pred_fine = trend_pred_fine.permute(0,2,1)
        #L2 regularization for trend prediction
        if self.fc_trend.weight.grad != None:
            self.fc_trend.weight.grad.data.add_(self.l2_fac * self.fc_trend.weight.data)
        return season, trend_pred_fine #season: [bs, seq_len, c_in], trend_pred_fine: [bs, pred_len, channel]


class Dlinear_decomp(nn.Module):
    """
    Dlinear decomposition
    """

    def __init__(self, seq_len, pred_len, c_in): #fc_dropout=0.0:
        super(Dlinear_decomp, self).__init__()
        class CfgDlinear:
            def __init__(self,seq_len,pred_len,enc_in):
                self.seq_len = seq_len
                self.pred_len = pred_len
                self.individual = False
                self.enc_in = enc_in
        self.model = DLinear.Model(CfgDlinear(seq_len, pred_len, c_in)).float()


    def forward(self, x):
        #x: [bs, seq_len, c_in]
        season=x
        trend_pred_fine = self.model(x)
        return season, trend_pred_fine #season: [bs, seq_len, c_in], trend_pred_fine: [bs, pred_len, c_in]

    
    
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, d_compress=None, layer_type=0,
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, reorder=False, **kwargs):
        #reorder: True for ordinary trasformer, False for reordered transformer
        #layer_type: 0 TSTEncoderLayer, 1 MappedTSTEncoderLayer, 2 MappedTSTCrossEncoderLayer, 
        #   3 RoutedTSTEncoderLayer, 4 LowRankTSTEncoderLayer, 5 MappedTSTEncoderLayerShared, 6 RoutedOriTSTEncoderLayer
        #d_compress: when layer_type==0, compress the q_len dim of k and v to d_compress, 0 to disable
        #            when layer_type==1,2, d_compress is the length of router 
        #d_compress is valid only when q_len is set

        super().__init__()

        if layer_type==0:
            layer=TSTEncoderLayer
        elif layer_type==1:
            layer=MappedTSTEncoderLayer
        elif layer_type==2:
            layer=MappedTSTCrossEncoderLayer
        elif layer_type==3:
            layer=RoutedTSTEncoderLayer
        elif layer_type==4:
            layer=LowRankTSTEncoderLayer
        elif layer_type==5:
            layer=MappedTSTEncoderLayerShared
        elif layer_type==6:
            layer=RoutedOriTSTEncoderLayer
        self.layers = nn.ModuleList([layer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      d_compress=d_compress,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn, reorder=reorder, **kwargs) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None, **kwargs):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask, **kwargs)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask, **kwargs)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, d_compress=None, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False, reorder=False):
        #reorder: True for ordinary trasformer, False for reordered transformer
        #d_compress: compress the q_len dim of k and v to d_compress, 0 to disable
        #d_compress is valid only when q_len is set

        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, 
                                             q_len=q_len, d_compress=d_compress, 
                                             attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention, reorder=reorder)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src



class RoutedTSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, d_compress=None, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu",  **kwargs):
        #d_compress: compress the q_len dim of k and v to d_compress, 0 to disable
        #d_v is not used

        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        d_k = d_model // n_heads if d_k is None else d_k
        self.q_len = q_len
        self.d_k = d_k
        self.d_v = d_v
        self.d_compress =  d_compress
        self.d_model = d_model
        self.d_model_attn = d_k*n_heads
        assert d_compress>0, "Error: d_compress<=0 in MappedTSTEncoderLayer"
        self.router = nn.Parameter(torch.randn([1,n_heads,d_compress,d_k]), requires_grad=True)
        self.W_K = nn.Linear(d_model, self.d_model_attn)
        self.W_V = nn.Linear(d_model, self.d_model_attn)
        self.to_out = nn.Linear(self.d_model_attn, d_model)
        # Scaled dot product attention
        self.sdp_attn = _ScaledDotProductAttention(self.d_model_attn, n_heads, attn_dropout=attn_dropout, res_attention=False, lsa=False)
        #Additional multi-head attention
        self.attn_2 = _MultiheadAttention(d_model, n_heads, d_k=d_k, d_v=d_k,
                                               attn_dropout=attn_dropout, proj_dropout=dropout)
        

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        self.d_ff = d_ff
        if d_ff != 0:
            # Position-wise Feed-Forward
            self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                    get_activation_fn(activation),
                                    nn.Dropout(dropout),
                                    nn.Linear(d_ff, d_model, bias=bias))

            # Add & Norm
            self.dropout_ffn = nn.Dropout(dropout)
            if "batch" in norm.lower():
                self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            else:
                self.norm_ffn = nn.LayerNorm(d_model)

        self.store_attn = store_attn


    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None, **kwargs) -> Tensor:

        ### Multi-Head attention sublayer
        #src: [bs, q_len, d_model]
        bs = src.shape[0]
        q_s = self.router.repeat(bs, 1, 1, 1)   # q_s: [bs, n_heads, d_compress, d_k]
        # k_s: [bs, n_heads, d_k, q_len]
        k_s = self.W_K(src).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     
        # v_s: [bs, n_heads, q_len, d_k]
        v_s = k_s.transpose(3,2) #TODOself.W_V(src).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)  
        #attn_router: [bs, n_heads, d_compress, d_k]     
        attn_router, _ = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        ## back to the original inputs dimensions
        #attn_router: [bs, d_compress, d_model_attn]
        attn_router = attn_router.transpose(1, 2).contiguous().view(bs, -1, self.d_model_attn) 
        #attn_router: [bs, d_compress, d_model]
        attn_router = self.to_out(attn_router)
        ## Additional attention
        src2, attn = self.attn_2(src, attn_router, attn_router, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        ##Add & Norm
        src = src + self.dropout_attn(src2) # Add residual connection with residual dropout
        src = self.norm_attn(src)
        if self.d_ff != 0:
            # Feed-forward sublayer
            ## Position-wise Feed-Forward
            src2 = self.ff(src)
            ## Add & Norm
            src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
            src = self.norm_ffn(src)
        return src
    


class MappedTSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, d_compress=None, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu",  **kwargs):
        #d_compress: compress the q_len dim of k and v to d_compress, 0 to disable
        #d_v is not used

        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        d_k = d_model // n_heads if d_k is None else d_k
        self.q_len = q_len
        self.d_k = d_k
        self.d_v = d_v
        self.d_compress =  d_compress
        self.d_model = d_model
        self.d_model_attn = d_k*n_heads
        assert d_compress>0, "Error: d_compress<=0 in MappedTSTEncoderLayer"
        self.router = nn.Parameter(torch.randn([1,n_heads,d_compress,d_k]), requires_grad=True)
        self.W_K = nn.Linear(d_model, self.d_model_attn)
        self.W_V = nn.Linear(d_model, self.d_model_attn)
        self.to_out = nn.Linear(self.d_model_attn, d_model)
        # Scaled dot product attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=False, lsa=False)
        self.m_expand = nn.Parameter(torch.randn([1,q_len, d_compress]), requires_grad=True)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        self.d_ff = d_ff
        if d_ff != 0:
            # Position-wise Feed-Forward
            self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                    get_activation_fn(activation),
                                    nn.Dropout(dropout),
                                    nn.Linear(d_ff, d_model, bias=bias))

            # Add & Norm
            self.dropout_ffn = nn.Dropout(dropout)
            if "batch" in norm.lower():
                self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            else:
                self.norm_ffn = nn.LayerNorm(d_model)

        self.store_attn = store_attn


    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None, **kwargs) -> Tensor:

        ### Multi-Head attention sublayer
        #src: [bs, q_len, d_model]
        bs = src.shape[0]
        q_s = self.router.repeat(bs, 1, 1, 1)   # q_s: [bs, n_heads, d_compress, d_k]
        # k_s: [bs, n_heads, d_k, q_len]
        k_s = self.W_K(src).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     
        # v_s: [bs, n_heads, q_len, d_k]
        v_s = k_s.transpose(3,2) #TODOself.W_V(src).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)  
        #attn_router: [bs, n_heads, d_compress, d_k]     
        attn_router, _ = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        ## back to the original inputs dimensions
        #attn_router: [bs, d_compress, d_model_attn]
        attn_router = attn_router.transpose(1, 2).contiguous().view(bs, -1, self.d_model_attn) 
        #attn_router: [bs, d_compress, d_model]
        attn_router = self.to_out(attn_router)
        ## Map & Add & Norm
        src2 = torch.matmul(self.m_expand, attn_router) #src2:  [bs, q_len, d_model]
        src = src + self.dropout_attn(src2) # Add residual connection with residual dropout
        src = self.norm_attn(src)
        if self.d_ff != 0:
            # Feed-forward sublayer
            ## Position-wise Feed-Forward
            src2 = self.ff(src)
            ## Add & Norm
            src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
            src = self.norm_ffn(src)
        return src



class LowRankTSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, d_compress=None, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu",  **kwargs):
        #d_compress: length of learnable router
        #d_v is not used

        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        d_k = d_model // n_heads if d_k is None else d_k
        self.q_len = q_len
        self.d_k = d_k
        self.d_compress =  d_compress
        self.d_model = d_model
        self.d_model_attn = d_k*n_heads
        assert d_compress>0, "Error: d_compress<=0 in MappedTSTEncoderLayer"
        self.router = nn.Parameter(torch.randn([1,n_heads,d_compress,d_k]), requires_grad=True)
        self.W_Q = nn.Linear(d_model, self.d_model_attn)
        self.to_out = nn.Linear(self.d_model_attn, d_model)
        # Scaled dot product attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=False, lsa=False)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        self.d_ff = d_ff
        if d_ff != 0:
            # Position-wise Feed-Forward
            self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                    get_activation_fn(activation),
                                    nn.Dropout(dropout),
                                    nn.Linear(d_ff, d_model, bias=bias))

            # Add & Norm
            self.dropout_ffn = nn.Dropout(dropout)
            if "batch" in norm.lower():
                self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            else:
                self.norm_ffn = nn.LayerNorm(d_model)

        self.store_attn = store_attn


    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None, **kwargs) -> Tensor:

        ### Multi-Head attention sublayer
        #src: [bs, q_len, d_model]
        bs = src.shape[0]
        #q_s: [bs*q_len, n_heads, 1, d_k]
        q_s = self.W_Q(src).view(bs*self.q_len, 1, self.n_heads, self.d_k).permute(0,2,1,3)
        #v_s: [bs, n_heads, d_compress, d_k]
        v_s = self.router.repeat(bs*self.q_len, 1, 1, 1)
        #k_s: [bs, n_heads, d_k, d_compress]
        k_s = v_s.transpose(3,2)      
        #attn_low_rank: [bs*q_len, n_heads, 1, d_k]     
        attn_low_rank, _ = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        ##Back to the original inputs dimensions
        #attn_low_rank: [bs*q_len, , d_model_attn]
        attn_low_rank = attn_low_rank.transpose(1, 2).contiguous().view(bs*self.q_len, 1, self.d_model_attn) 
        attn_low_rank = self.to_out(attn_low_rank)  #attn_low_rank: [bs*q_len, 1, d_model]
        ##Add & Norm
        src = src.reshape(bs*self.q_len, 1, self.d_model)
        src = src + self.dropout_attn(attn_low_rank) # Add residual connection with residual dropout
        src = self.norm_attn(src)
        if self.d_ff != 0:
            # Feed-forward sublayer
            ## Position-wise Feed-Forward
            src2 = self.ff(src)
            ## Add & Norm
            src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
            src = self.norm_ffn(src)
        src = src.reshape(bs, self.q_len, self.d_model)
        return src    


class MappedTSTEncoderLayerShared(nn.Module):
    def __init__(self, q_len, d_model, n_heads, n_tf:int=0, d_k=None, d_v=None, d_ff=256, d_compress=None, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", 
                 b_q_proj:bool=False, b_share_kv:bool=True, b_compress_pos_emb:bool=True, **kwargs):
        #d_compress: compress the q_len dim of k and v to d_compress, 0 to disable
        #d_v is not used
        #n_tf:  n_t+n_f, set to 0 to disable tf scaling

        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        d_k = d_model // n_heads if d_k is None else d_k
        self.q_len = q_len
        self.d_k = d_k
        self.d_v = d_v
        self.d_compress =  d_compress
        self.d_model = d_model
        self.d_model_attn = d_k*n_heads
        self.n_tf = n_tf
        
        self.b_q_proj = b_q_proj
        assert d_compress>0, "Error: d_compress<=0 in MappedTSTEncoderLayer"
        if not self.b_q_proj:
            self.router = nn.Parameter(torch.randn([1,n_heads,d_compress,d_k]), requires_grad=True)
        else:
            print('Ablation study, projection of Q enabled in LRA')
            self.router = nn.Parameter(torch.randn([1,d_compress,d_model]), requires_grad=True)
            self.W_Q = nn.Linear(d_model, self.d_model_attn)
        self.W_K = nn.Linear(d_model, self.d_model_attn)
        
        self.b_share_kv = b_share_kv
        if not b_share_kv:
            print('Ablation study, donot share the projection of K and V in LRA')
            self.W_V = nn.Linear(d_model, self.d_model_attn)
        self.to_out = nn.Linear(self.d_model_attn, d_model)
        # Scaled dot product attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=False, lsa=False)
        self.m_expand = nn.Parameter(torch.randn([1,q_len, d_compress]), requires_grad=True)
        # Position embedding on the compressed attention
        self.b_compress_pos_emb=b_compress_pos_emb
        if b_compress_pos_emb:
            self.pos_embd = nn.Parameter(torch.zeros([1,1,d_compress,d_model]), requires_grad=True)
        else:
            print('Ablation study, vanilla position embedding in LRA')
            self.pos_embd = nn.Parameter(torch.zeros([1,1,q_len,d_model]), requires_grad=True)
        #tf scaling
        if self.n_tf >0:
            self.scale_tf = nn.Parameter(torch.ones([1,self.n_tf,1,1]), requires_grad=True)
        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        self.d_ff = d_ff
        if d_ff != 0:
            # Position-wise Feed-Forward
            self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                    get_activation_fn(activation),
                                    nn.Dropout(dropout),
                                    nn.Linear(d_ff, d_model, bias=bias))

            # Add & Norm
            self.dropout_ffn = nn.Dropout(dropout)
            if "batch" in norm.lower():
                self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            else:
                self.norm_ffn = nn.LayerNorm(d_model)

        self.store_attn = store_attn


    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None, **kwargs) -> Tensor:

        ### Multi-Head attention sublayer
        #src: [bs, d_share_corr, q_len, d_model], d_share_corr is n_freq+n_concat_td in our application
        bs = src.shape[0]
        d_share_corr = src.shape[1]
        if self.n_tf>0 and self.n_tf != d_share_corr:
            assert 0, "Error: n_tf != d_share_corr in MappedTSTEncoderLayer"
        q_s = self.router.repeat(bs, 1, 1, 1)   # q_s: [bs, n_heads, d_compress, d_k]
        if self.b_q_proj: # q_s: in [bs,d_compress,d_model]
            q_s = self.W_Q(q_s).view(bs, self.d_compress, self.n_heads, self.d_k)
            q_s = q_s.permute(0,2,1,3)
        #Vanilla position embedding
        if not self.b_compress_pos_emb:
            src = src + self.pos_embd
        # k_s: [bs, n_heads, d_k, q_len*d_share_corr]
        k_s = self.W_K(src).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     
        # v_s: [bs, n_heads, q_len*d_share_corr, d_k]
        if self.b_share_kv:
            v_s = k_s.transpose(3,2) 
        else:
            v_s = self.W_V(src).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)  
        #attn_router: [bs, n_heads, d_compress, d_v]     
        attn_router, _ = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        ## back to the original inputs dimensions
        #attn_router: [bs, d_compress, d_model_attn]
        attn_router = attn_router.transpose(1, 2).contiguous().view(bs, -1, self.d_model_attn) 
        #attn_router: [bs, 1, d_compress, d_model]
        attn_router = self.to_out(attn_router).reshape(bs, 1, self.d_compress, self.d_model)
        ## Position embedding
        if self.b_compress_pos_emb:
            attn_router = attn_router + self.pos_embd 
        ## Map & Add & Norm
        src2 = torch.matmul(self.m_expand, attn_router) #src2:  [bs, 1, q_len, d_model]
        if self.n_tf>0:
            src2 = src2 * self.scale_tf #src2:  [bs, n_tf, q_len, d_model]
        src = src + self.dropout_attn(src2) # Add residual connection with residual dropout
        src = src.reshape(bs*d_share_corr, self.q_len, self.d_model)
        src = self.norm_attn(src)
        if self.d_ff != 0:
            # Feed-forward sublayer
            ## Position-wise Feed-Forward
            src2 = self.ff(src)
            ## Add & Norm
            src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
            src = self.norm_ffn(src)
        src = src.reshape(bs, d_share_corr, self.q_len, self.d_model)
        return src    


class MappedTSTCrossEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, d_compress=None, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu",  **kwargs):
        #d_compress: compress the q_len dim of k and v to d_compress, 0 to disable
        #d_v is not used

        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        d_k = d_model // n_heads if d_k is None else d_k
        self.q_len = q_len
        self.d_k = d_k
        self.d_v = d_v
        self.d_compress =  d_compress
        self.d_model = d_model
        self.d_model_attn = d_k*n_heads
        assert d_compress>0, "Error: d_compress<=0 in MappedTSTEncoderLayer"
        self.router = nn.Parameter(torch.randn([1,n_heads,d_compress,d_k]), requires_grad=True)
        self.W_K = nn.Linear(d_model, self.d_model_attn)
        self.W_V = nn.Linear(d_model, self.d_model_attn)
        self.to_out = nn.Linear(self.d_model_attn, d_model)
        # Scaled dot product attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=False, lsa=False)
        self.m_expand = nn.Parameter(torch.randn([1,q_len, d_compress]), requires_grad=True)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        self.d_ff = d_ff
        if d_ff != 0:
            # Position-wise Feed-Forward
            self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                    get_activation_fn(activation),
                                    nn.Dropout(dropout),
                                    nn.Linear(d_ff, d_model, bias=bias))

            # Add & Norm
            self.dropout_ffn = nn.Dropout(dropout)
            if "batch" in norm.lower():
                self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            else:
                self.norm_ffn = nn.LayerNorm(d_model)

        self.store_attn = store_attn


    def forward(self, src:Tensor, src_cross_in:Tensor, src_cross_out:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None, **kwargs) -> Tensor:

        ### Multi-Head attention sublayer
        #src: [bs, d_share_corr, q_len, d_model], src_cross_in: [bs, q_len_cross, d_model], d_share_corr is n_freq+n_concat_td in our application
        bs = src_cross_in.shape[0]
        d_share_corr = src.shape[1]
        q_s = self.router.repeat(bs, 1, 1, 1)   # q_s: [bs, n_heads, d_compress, d_k]
        # k_s_in: [bs, n_heads, d_k, q_len_cross_in]
        k_s_in = self.W_K(src_cross_in).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     
        # v_s_in: [bs, n_heads, q_len_cross_in, d_k]
        v_s_in = k_s_in.transpose(3,2) 
        #attn_router_in: [bs, n_heads, d_compress, d_k]     
        attn_router_in, _ = self.sdp_attn(q_s, k_s_in, v_s_in, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        ## back to the original inputs dimensions
        #attn_router_in: [bs, d_compress, d_model_attn]
        attn_router_in = attn_router_in.transpose(1, 2).contiguous().view(bs, -1, self.d_model_attn)
        # k_s_out: [bs, n_heads, d_k, q_len_cross_out]
        k_s_out = self.W_K(src_cross_out).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     
        # v_s_out: [bs, n_heads, q_len_cross_out, d_k]
        v_s_out = k_s_out.transpose(3,2) 
        #attn_router_out: [bs, n_heads, d_compress, d_k]     
        attn_router_out, _ = self.sdp_attn(q_s, k_s_out, v_s_out, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        ## back to the original inputs dimensions
        #attn_router_out: [bs, d_compress, d_model_attn]
        attn_router_out = attn_router_out.transpose(1, 2).contiguous().view(bs, -1, self.d_model_attn)
        attn_router = torch.layer_norm(attn_router_in * attn_router_out, (self.d_model_attn,))  
        # attn_router: [bs, d_compress, d_model]
        attn_router = self.to_out(attn_router)
        ## Map & Add & Norm
        src2 = torch.matmul(self.m_expand, attn_router) #src2:  [bs, q_len, d_model]
        src2 = src2.reshape(bs, 1, self.q_len, self.d_model)
        src = src + self.dropout_attn(src2) # Add residual connection with residual dropout
        src = src.reshape(bs*d_share_corr, self.q_len, self.d_model)
        src = self.norm_attn(src)
        if self.d_ff != 0:
            # Feed-forward sublayer
            ## Position-wise Feed-Forward
            src_ff = self.ff(src)
            ## Add & Norm
            src = src + self.dropout_ffn(src_ff) # Add: residual connection with residual dropout
            src = self.norm_ffn(src)
        src = src.reshape(bs, d_share_corr, self.q_len, self.d_model)
        return src


#Routed layer of Crossformer
class RoutedOriTSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, d_compress=None, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu",  **kwargs):
        #d_compress: length of the router
        #d_v is not used

        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        d_k = d_model // n_heads if d_k is None else d_k
        self.q_len = q_len
        self.d_k = d_k
        self.d_v = d_v
        self.d_compress =  d_compress
        self.d_model = d_model
        self.d_model_attn = d_k*n_heads
        assert d_compress>0, "Error: d_compress<=0 in RoutedOriTSTEncoderLayer"
        self.router = nn.Parameter(torch.randn([1,d_compress,d_model]), requires_grad=True)
        #Multi-head attentions
        self.attn_1 = _MultiheadAttention(d_model, n_heads, d_k=d_k, d_v=d_k,
                                                attn_dropout=attn_dropout, proj_dropout=dropout)
        self.attn_2 = _MultiheadAttention(d_model, n_heads, d_k=d_k, d_v=d_k,
                                               attn_dropout=attn_dropout, proj_dropout=dropout)
        

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        self.d_ff = d_ff
        if d_ff != 0:
            # Position-wise Feed-Forward
            self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                    get_activation_fn(activation),
                                    nn.Dropout(dropout),
                                    nn.Linear(d_ff, d_model, bias=bias))

            # Add & Norm
            self.dropout_ffn = nn.Dropout(dropout)
            if "batch" in norm.lower():
                self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            else:
                self.norm_ffn = nn.LayerNorm(d_model)

        self.store_attn = store_attn


    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None, **kwargs) -> Tensor:

        ### Multi-Head attention sublayer
        #src: [bs, share_len, q_len, d_model]
        bs, share_len, q_len, d_model = src.shape
        q_s = self.router.repeat(bs*share_len, 1, 1)   # q_s: [bs, d_compress, d_model]
        src=src.reshape(bs*share_len,q_len, d_model)
        attn_router, _ = self.attn_1(q_s, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        ## Additional attention
        src2, _ = self.attn_2(src, attn_router, attn_router, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        ##Add & Norm
        src = src + self.dropout_attn(src2) # Add residual connection with residual dropout
        src = self.norm_attn(src)
        if self.d_ff != 0:
            # Feed-forward sublayer
            ## Position-wise Feed-Forward
            src2 = self.ff(src)
            ## Add & Norm
            src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
            src = self.norm_ffn(src)
        src=src.reshape(bs,share_len,q_len, d_model)
        return src



class MixedTSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, c_in=None, d_k=None, d_v=None, d_ff=256, d_compress=None, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", b_tf_2_bs=False, **kwargs):
        #c_in: number of channels
        #b_tf_2_bs: merge time-freq dimension to batchsize

        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        d_k = d_model // n_heads if d_k is None else d_k
        self.q_len = q_len
        self.d_k = d_k
        self.d_v = d_v
        self.d_compress =  d_compress
        self.d_model = d_model
        self.d_model_attn = d_k*n_heads
        assert d_compress>0, "Error: d_compress<=0 in MixedTSTEncoderLayer"
        assert c_in > 0, "Error: n_c<=0 in MixedTSTEncoderLayer"
        self.c_in = c_in
        self.b_tf_2_bs = b_tf_2_bs

        self.router = nn.Parameter(torch.randn([1,n_heads,d_compress,d_k]), requires_grad=True)
        self.W_Q = nn.Linear(d_model, self.d_model_attn)
        self.W_K = nn.Linear(d_model, self.d_model_attn)
        self.W_V = nn.Linear(d_model, self.d_model_attn)
        self.to_out = nn.Linear(self.d_model_attn, d_model)
        # Scaled dot product attention
        self.sdp_attn_tf = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=False, lsa=False)
        self.sdp_attn_chan = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=False, lsa=False)
        self.m_expand = nn.Parameter(torch.randn([1,c_in, d_compress]), requires_grad=True)
        
        # Position embedding for the compressed attention
        self.pos_embd = nn.Parameter(torch.zeros([1,d_compress,self.d_model_attn]), requires_grad=True)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        self.d_ff = d_ff
        if d_ff != 0:
            # Position-wise Feed-Forward
            self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                    get_activation_fn(activation),
                                    nn.Dropout(dropout),
                                    nn.Linear(d_ff, d_model, bias=bias))

            # Add & Norm
            self.dropout_ffn = nn.Dropout(dropout)
            if "batch" in norm.lower():
                self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            else:
                self.norm_ffn = nn.LayerNorm(d_model)

        self.store_attn = store_attn


    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None, **kwargs) -> Tensor:

        ### Multi-Head attention sublayer
        #src: [bs*c_in, n_freq+n_concat_td, d_model]
        bs = src.shape[0]//self.c_in
        n_tf = self.q_len #n_tf=n_freq+n_concat_td
        #Attention along the time-frequecy direction
        #q_s_tf: [bs*c_in, n_heads, n_tf, d_k]
        q_s_tf = self.W_Q(src).view(bs*self.c_in, n_tf, self.n_heads, self.d_k).permute(0,2,1,3) 
        #k_s_tf: [bs*c_in, n_heads, d_k, n_tf]
        k_s_tf = self.W_K(src).view(bs*self.c_in, n_tf, self.n_heads, self.d_k).permute(0,2,3,1) 
        #v_s_tf: [bs*c_in, n_heads, n_tf, d_k]
        v_s_tf = self.W_V(src).view(bs*self.c_in, n_tf, self.n_heads, self.d_k).permute(0,2,1,3) 
        #attn_tf: [bs*c_in, n_heads, n_tf, d_k]
        attn_tf, _ = self.sdp_attn_tf(q_s_tf, k_s_tf, v_s_tf, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        attn_tf = attn_tf.permute(0,2,1,3).reshape(bs*self.c_in, n_tf, self.d_model_attn)

        #Attention along the channel direction
        if not self.b_tf_2_bs:
            #q_s_c: [bs, n_heads, d_compress, d_k]
            q_s_c = self.router.repeat(bs, 1, 1, 1)   
            #k_s_c: [bs, n_heads, d_k, c_in*n_tf]
            k_s_c = k_s_tf.reshape(bs, self.c_in, self.n_heads, self.d_k, n_tf)
            k_s_c = k_s_c.permute(0,2,3,1,4)
            k_s_c = k_s_c.reshape(bs, self.n_heads, self.d_k, self.c_in*n_tf)
            #v_s_c: [bs, n_heads, c_in*n_tf, d_k]
            v_s_c = v_s_tf.reshape(bs, self.c_in, self.n_heads, n_tf, self.d_k)
            v_s_c = v_s_c.permute(0,2,3,1,4)
            v_s_c = v_s_c.reshape(bs, self.n_heads, n_tf*self.c_in, self.d_k)
            #attn_chan: [bs, n_heads, d_compress, d_k]
            attn_chan, _ = self.sdp_attn_chan(q_s_c, k_s_c, v_s_c, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            attn_chan = attn_chan.transpose(1,2).reshape(bs, self.d_compress, self.d_model_attn)
            #Position embedding
            attn_chan = attn_chan + self.pos_embd #attn_chan: [bs, d_compress, d_model_attn]
            #Expand. m_expand: [1,c_in, d_compress]
            attn_chan = torch.matmul(self.m_expand, attn_chan) #attn_chan: [bs, c_in, d_model_attn]
            attn_chan = attn_chan.reshape(bs*self.c_in, 1, self.d_model_attn)
        else:
            #q_s_c: [bs*n_tf, n_heads, d_compress, d_k]
            q_s_c = self.router.repeat(bs*n_tf, 1, 1, 1)   
            #k_s_c: [bs*n_tf, n_heads, d_k, c_in]
            k_s_c = k_s_tf.reshape(bs, self.c_in, self.n_heads, self.d_k, n_tf)
            k_s_c = k_s_c.permute(0,4,2,3,1) #[bs, n_tf, n_heads, d_k, c_in]
            k_s_c = k_s_c.reshape(bs*n_tf, self.n_heads, self.d_k, self.c_in)
            #v_s_c: [bs*n_tf, n_heads, c_in, d_k]
            v_s_c = v_s_tf.reshape(bs, self.c_in, self.n_heads, n_tf, self.d_k)
            v_s_c = v_s_c.permute(0,3,2,1,4) #[bs, n_tf, n_heads, c_in, d_k]
            v_s_c = v_s_c.reshape(bs*n_tf, self.n_heads, self.c_in, self.d_k)
            #attn_chan: [bs*n_tf, n_heads, d_compress, d_k]
            attn_chan, _ = self.sdp_attn_chan(q_s_c, k_s_c, v_s_c, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            attn_chan = attn_chan.transpose(1,2).reshape(bs*n_tf, self.d_compress, self.d_model_attn)
            #Position embedding
            attn_chan = attn_chan + self.pos_embd #attn_chan: [bs*n_tf, d_compress, d_model_attn]
            #Expand. m_expand: [1,c_in, d_compress]
            attn_chan = torch.matmul(self.m_expand, attn_chan) #attn_chan: [bs*n_tf, c_in, d_model_attn]
            attn_chan = attn_chan.reshape(bs,n_tf,self.c_in, self.d_model_attn)
            attn_chan = attn_chan.transpose(1,2).reshape(bs*self.c_in, n_tf, self.d_model_attn)

        #Merge attentions
        attn_merged = attn_tf+attn_chan
        attn_out = self.to_out(attn_merged) #attn_out: [bs*c_in, n_tf, d_model]

        #Add & Norm & Position-wise Feed-Forward
        src = src + self.dropout_attn(attn_out) # Add residual connection with residual dropout
        src = self.norm_attn(src)
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        src = self.norm_ffn(src)
        return src   #src: [bs*c_in, n_freq+n_concat_td, d_model]
    


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, q_len:Optional[int]=None, d_compress=None, res_attention=False, reorder=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs), max_q_len, d_model]
            K, V:    [batch_size (bs), q_len, d_model]
            mask:    [q_len, q_len]
        reorder: True for ordinary trasformer, False for reordered transformer
        d_compress: compress the q_len dim of k and v to d_compress, 0 to disable
        d_compress is valid only when q_len is set
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        if d_compress:
            assert q_len,"Error: d_compress is valid only when q_len is set"
            self.W_compress = nn.Linear(q_len, d_compress, bias=False)
        self.d_compress = d_compress
        self.q_len = q_len

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        if not reorder: 
            self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)
        else:
            self.sdp_attn = _ReorderScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention)
        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs, n_heads, max_q_len, d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs, n_heads, d_k, q_len] - transpose(1,2) + transpose(2,3)
        if self.d_compress is None:
            v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs, n_heads, q_len, d_v]
        else:
            k_s = self.W_compress(k_s)     # k_s    [bs, n_heads, d_k, d_compress]    
            v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).permute(0,2,3,1) 
            v_s = self.W_compress(v_s)     # v_s(out)    [bs, n_heads, d_k, d_compress] 
            v_s = v_s.permute(0,1,3,2)     # v_s(out)    [bs, n_heads, d_compress, d_k]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs, n_heads, q_len, d_v], attn: [bs, n_heads, q_len, q_len], scores: [bs, n_heads, max_q_len, q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs, q_len, n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs, n_heads, max_q_len, d_k]
            k               : [bs, n_heads, d_k, seq_len]
            v               : [bs, n_heads, seq_len, d_v]
            prev            : [bs, n_heads, q_len, seq_len]
            key_padding_mask: [bs, seq_len]
            attn_mask       : [1, seq_len, seq_len]
        Output shape:
            output:  [bs, n_heads, q_len, d_v]
            attn   : [bs, n_heads, q_len, seq_len]
            scores : [bs, n_heads, q_len, seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs, n_heads, max_q_len, q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len, seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs, q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs, n_heads, max_q_len, q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs, n_heads, max_q_len, d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ReorderScaledDotProductAttention(nn.Module):
    r"""Reordered Scaled Dot-Product Attention module"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs, n_heads, max_q_len, d_k]
            k               : [bs, n_heads, d_k, seq_len]
            v               : [bs, n_heads, seq_len, d_v]
            prev            : [bs, n_heads, d_k, d_v]
            attn_mask       : [1, seq_len, seq_len]
        Output shape:
            output:  [bs, n_heads, q_len, d_v]
            attn   : [bs, n_heads, q_len, seq_len]
            scores : [bs, n_heads, q_len, seq_len]
        key_padding_mask is ignored in this version
        '''

        seq_len = v.shape[2]

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(k, v) / seq_len**0.5      # attn_scores : [bs, n_heads, d_k, d_v]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [d_k, d_v] - only used when d_k == d_v
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=0)                 # attn_weights   : [bs, n_heads, d_k, d_v]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(q, attn_weights)                        # output: [bs, n_heads, max_q_len, d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights
