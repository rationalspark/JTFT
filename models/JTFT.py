# This file is licensed under the GNU General Public License (GPL) version 2.0.
# See the LICENSE file or https://www.gnu.org/licenses/gpl-2.0.html for more details.

__all__ = ['JTFT']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.FreqTST_backbone import *
from layers.PatchTST_layers import *

class Model(nn.Module):
    def __init__(self, configs, 
                 b_aux_head:bool=True,
                 max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        self.c_in = configs.enc_in
        context_window = configs.seq_len
        self.target_window = configs.pred_len
        self.seq_len = configs.seq_len 
        n_concat_td = configs.n_concat_td
        self.n_decomp = configs.decomposition
        padding_patch = configs.padding_patch
        d_compress_max = configs.d_compress_max
        mod_scal_tfi = configs.mod_scal_tfi
        use_mark = configs.use_mark
        
        n_layers = configs.e_layers
        n_layers_tfi = configs.e_layers_tfi
        if n_layers_tfi == None:
            n_layers_tfi = n_layers 
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout    
        self.patch_len = configs.patch_len
        n_freq = configs.n_freq
        self.stride = configs.stride
        self.b_learn_freq = True
        assert context_window % self.stride == 0, "Error: context_window % stride != 0"

        if hasattr(configs, 'fd_analysis'):
        #Analysis data and do not predict
            self.model=FreqTST_reconstuct(c_in=self.c_in, n_freq=n_freq, 
                                          context_window=self.seq_len, b_learn_freq=False)
            return
            

        #Initialize model
        
        self.model=FreqTST_ci_tfi(c_in=self.c_in,
                n_freq=n_freq, n_concat_td=n_concat_td,
                context_window=self.seq_len, target_window=self.target_window, 
                mod_scal_tfi = mod_scal_tfi,
                patch_len=self.patch_len, stride=self.stride, 
                d_compress_max=d_compress_max, 
                use_mark=use_mark,
                sep_time_freq=configs.sep_time_freq,
                b_learn_freq = self.b_learn_freq,
                n_decomp = self.n_decomp, #b_ori_router=True,
                n_layers=n_layers,  n_layers_tfi= n_layers_tfi, d_model=d_model, n_heads=n_heads,d_ff=d_ff, 
                dropout=dropout, fc_dropout=fc_dropout, head_dropout = head_dropout, padding_patch=padding_patch, **kwargs)
        
       

                
    def forward(self, x, z_mark, target_mark, **kwargs):           # x: [bs x seq_len x channel]
        x = x[:, -self.seq_len:, :] #Get input for FD model
        #Call CI_TFI model. out_mod [bs x pred_len x channel]
        out_mod = self.model(x, z_mark=z_mark, target_mark=target_mark, **kwargs)
        return out_mod
    

                


    
