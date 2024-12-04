import copy
from typing import Optional, Any, Union, Callable

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.nn.init as init

from utils.datasets import separate_features_by_index

class nnMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, fix_baseline_risk=-1):
        super().__init__()
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, num_labels),
            # nn.Flatten(0, 1)
        )
        self.num_labels = num_labels
        self.fix_baseline_risk = fix_baseline_risk
        self.reset_params()
    
    def reset_params(self):
        W1, b1 = self.mlp[0].weight.data, self.mlp[0].bias.data
        W2, b2 = self.mlp[2].weight.data, self.mlp[2].bias.data
        W3, b3 = self.mlp[4].weight.data, self.mlp[4].bias.data

        gamma_dist = dist.Gamma(1, 100)
        W1 = gamma_dist.sample(W1.shape)
        W2 = gamma_dist.sample(W2.shape)
        W3 = gamma_dist.sample(W3.shape)
        # nn.init.ones_(self.mlp[4].weight)
        b1 = -gamma_dist.sample(b1.shape)
        b2 = -gamma_dist.sample(b2.shape)
        if self.fix_baseline_risk > 0:
            b3 = torch.reshape(torch.tensor(self.fix_baseline_risk), b3.shape)
        else:
            b3 = gamma_dist.sample(b3.shape)

        self.mlp[0].weight.data, self.mlp[0].bias.data = W1, b1
        self.mlp[2].weight.data, self.mlp[2].bias.data = W2, b2
        # self.mlp[4].bias.data = b3
        self.mlp[4].weight.data, self.mlp[4].bias.data = W3, b3
    
    def enforce_positive_weights(self):
        # Constrain the weights to be positive
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.weight.data.clamp_(min=0, max=None)

                # Constrain all (except last layer's) biases to be negative
                if layer is not self.mlp[-1]:
                    layer.bias.data.clamp_(min=None, max=0)
        
    def freeze_weights(self):
        self.mlp[4].weight.requires_grad = False

    def forward(self, x):
        x = self.flatten(x)
        logits = self.mlp(x)
        return logits
    
class num_mlp(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, hidden_size*input_size),
        )
        self.reset_params()
    
    def reset_params(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                init.normal_(layer.weight, mean=0, std=0.02)
                init.constant_(layer.bias, 0)
    
    def enforce_positive_weights(self):
        # Constrain the weights to be positive (optional property during training)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.weight.data.clamp_(min=0)

    def forward(self, x):
        B = x.size(0)
        x = self.flatten(x)
        logits = self.mlp(x)
        return logits.reshape(B, self.input_size, -1)

class LinearClassifier(nn.Module):
    """Linear layer to train on top of transformer network"""
    def __init__(self, dim, num_labels=1):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        nn.init.constant_(self.linear.bias, 0)
    
    def enforce_positive_weights(self):
        self.linear.weight.data.clamp_(min=0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)

class TransformerEncoderModel(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, mlp_ratio, dropout=0.0):
        super().__init__()

        self.cls_token = nn.Parameter(torch.rand(1, 1, d_model))
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model, num_heads, mlp_ratio*d_model, dropout, activation='gelu', batch_first=True),
            num_layers
        )

    def forward(self, x, return_attn=False):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if return_attn:
            x, attn = self.transformer_encoder(x, return_attn=True)
            return x, attn
        else:
            return self.transformer_encoder(x)

class TRACE(nn.Module):
    def __init__(self, 
                 hidden_size, 
                 feature_metadata,
                 num_indices,
                 num_mode='mlp',
                 num_labels=1, 
                 dropout_p=0.0, 
                 cls_token=False, 
                 tran_layers=1, 
                 heads=2, 
                 mlp_ratio=4,
                 use_num_norm=False,
                 use_cat_norm=False,
                 checkbox_mode='sum'):
        super().__init__()

        self.hidden_size = hidden_size
        
        self.feature_metadata = feature_metadata
        self.feature_metadata_values_list = list(feature_metadata['categorical'].values())
        
        self.numerical_count = len(feature_metadata['continuous']) # Number of numerical/continuous features inside the dataset
        self.num_indices = num_indices
        self.num_mode = num_mode

        self.num_labels = num_labels
        self.cls_token = cls_token

        # Transformer Encoder Parameters
        self.tran_layers = tran_layers
        self.heads = heads
        self.mlp_ratio = mlp_ratio

        self.checkbox_mode = checkbox_mode
        self.embed_sizes = self._get_embed_input_size()

        # FIXME: checkbox data as 0-1 or as the others with 0 missing values
        self.embeddings = nn.ModuleList([
                            nn.ModuleList([nn.Embedding(subi, self.hidden_size) for subi in i]) if isinstance(i, list)
                            else nn.Embedding(i+1, self.hidden_size, padding_idx=0) for i in self.embed_sizes])
        self._embeddings_init()
        
        self.use_cat_norm = use_cat_norm
        if self.use_cat_norm:
            self.cat_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)

        self.use_num_norm = use_num_norm
        self.norm_tokens = 0
        if self.numerical_count > 0:
            if self.num_mode == 'mlp':
                self.num_mlp = num_mlp(self.numerical_count, self.hidden_size)
                if self.use_num_norm:
                    self.norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
            elif self.num_mode == 'norm':
                self.norm = nn.LayerNorm(self.numerical_count, eps=1e-6)
                self.norm_tokens = self.numerical_count
            elif self.num_mode == 'embed':
                self.num_embeddings = nn.Parameter(torch.rand(1, self.numerical_count, self.hidden_size))
            else:
                raise ValueError('Unknown numerical mode handling')

        self.dropout = nn.Dropout(dropout_p)

        self.encoder = TransformerEncoderModel(d_model=self.hidden_size, 
                                               num_layers=self.tran_layers, 
                                               num_heads=self.heads, 
                                               mlp_ratio=self.mlp_ratio, 
                                               dropout=dropout_p)
        
        self.head = LinearClassifier(self.hidden_size + self.norm_tokens, self.num_labels)
    
    def _embeddings_init(self):
        for e in self.embeddings:
            if isinstance(e, nn.Embedding):
                nn.init.kaiming_uniform_(e.weight, nonlinearity='relu')  # He initialization
            elif isinstance(e, nn.ModuleList):
                for sube in e:
                    if isinstance(sube, nn.Embedding):
                        nn.init.kaiming_uniform_(sube.weight, nonlinearity='relu')  # He initialization

    def _get_embed_input_size(self):
        token_sizes = []
        for t in self.feature_metadata_values_list:
            if isinstance(t, list):
                t = [subt+1 if subt == 1 else subt for subt in t]
                token_sizes.append(t)
            elif t == 1:
                token_sizes.append(t+1)
            else:
                token_sizes.append(t)
        return token_sizes
    
    def tokenize_input(self, x):
        tokens = []
        start = 0
        for size in self.feature_metadata_values_list:
            if isinstance(size, list):
                subtokens = []
                for _ in size:
                    subtokens.append(x[:, start:start+1])
                    start += 1
                tokens.append(subtokens)
            else:
                tokens.append(x[:, start:start+1])
                start += 1
        return tokens
    
    def calc_embeddings(self, x_cat, checkbox_mode='sum'):
        embedded_list = []
        # Loop through tokens
        for i, token in enumerate(x_cat):
            if isinstance(token, list):
                x_embedded = torch.stack([self.embeddings[i][j](subtoken) for j, subtoken in enumerate(token)], dim=1).squeeze()
                mask = torch.cat(token, dim=-1).unsqueeze(-1)
                x_embedded = x_embedded * mask
                
                if checkbox_mode == 'sum':
                    x_embedded = torch.sum(x_embedded, dim=1, keepdim=True)   
                elif checkbox_mode == 'prod':
                    x_embedded = torch.prod(x_embedded, dim=1, keepdim=True)
                elif checkbox_mode == 'max':
                    x_embedded, _ = torch.max(x_embedded, dim=1, keepdim=True)
                else:
                    raise ValueError(f"Unsupported interaction mode '{checkbox_mode}'. Supported modes are: 'sum', 'prod', 'max'.")
            else:    
                x_embedded = self.embeddings[i](token)
            
            embedded_list.append(x_embedded)
        
        return torch.cat(embedded_list, dim=1) # (n_samples, n_tokens, hidden_size)
    
    def forward(self, input, return_attn=False):
        B = input.shape[0]
        
        x_cat, x_num = separate_features_by_index(input, self.num_indices, num_dtype=torch.float32, cat_dtype=torch.int32)
        
        x_cat = self.tokenize_input(x_cat)

        # Pass categorical features through the embeddings
        x = self.calc_embeddings(x_cat, checkbox_mode=self.checkbox_mode)
        if self.use_cat_norm:
            x = self.cat_norm(x)
        x = self.dropout(x)
        
        # Numerical feature handling
        if self.numerical_count > 0:
            if self.num_mode == 'embed':
                num_embeddings = self.num_embeddings.expand(B, -1, -1)
                x_num = x_num.unsqueeze(-1).expand(-1, -1, self.hidden_size)
                x_num = x_num * num_embeddings
                x = torch.cat((x_num, x), dim=1)
            elif self.num_mode == 'mlp':
                x_num_weights = self.num_mlp(x_num)
                if self.use_num_norm:
                    x_num_weights = self.norm(x_num_weights)
                x_num_mask = (x_num != 0.).int()
                x_num_weights = x_num_mask.unsqueeze(-1) * x_num_weights
                x = torch.cat((x_num_weights, x), dim=1)
        
        if return_attn:
            x, attn = self.encoder(x, return_attn=True)
        else:
            x = self.encoder(x)
        
        if self.cls_token:
            x = x[:, 0]
        else:
            x = x[:, 1:]
            x = x.mean(dim=1)
        
        if self.numerical_count > 0 and self.num_mode == 'norm':
            x_num = self.norm(x_num) 
            x = torch.cat((x_num, x), dim=-1)

        if return_attn:
            return self.head(x), attn
        else:
            return self.head(x)

class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check

    def forward(
            self,
            src: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None,
            is_causal: Optional[bool] = None, 
            return_attn: bool = False) -> torch.Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            is_causal: If specified, applies a causal mask as mask (optional)
                and ignores attn_mask for computing scaled dot product attention.
                Default: ``False``.
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        if not isinstance(first_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{str_first_layer} was not TransformerEncoderLayer"
        elif first_layer.norm_first :
            why_not_sparsity_fast_path = f"{str_first_layer}.norm_first was True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not first_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = f" {str_first_layer}.self_attn.batch_first was not True"
        elif not first_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = f"{str_first_layer}.self_attn._qkv_same_embed_dim was not True"
        elif not first_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = f" {str_first_layer}.activation_relu_or_gelu was not True"
        elif not (first_layer.norm1.eps == first_layer.norm2.eps) :
            why_not_sparsity_fast_path = f"{str_first_layer}.norm1.eps was not equal to {str_first_layer}.norm2.eps"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif not self.enable_nested_tensor:
            why_not_sparsity_fast_path = "enable_nested_tensor was not True"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
                and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif first_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )

            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not (src.is_cuda or 'cpu' in str(src.device)):
                why_not_sparsity_fast_path = "src is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        # Prevent type refinement
        make_causal = (is_causal is True)

        if is_causal is None:
            if mask is not None:
                sz = mask.size(0)
                causal_comparison = torch.triu(
                    torch.ones(sz, sz, device=mask.device) * float('-inf'), diagonal=1
                ).to(mask.dtype)

                if torch.equal(mask, causal_comparison):
                    make_causal = True

        is_causal = make_causal

        for mod in self.layers:
            if return_attn:
                output, attn = mod(output, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers, return_attn=return_attn)
            else:
                output = mod(output, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers)

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = self.norm(output)

        if return_attn:
            return output, attn
        else:
            return output
    
class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation described in
        `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`_ if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.

        .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135

    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None,
            is_causal: bool = False, 
            return_attn: bool = False) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            is_causal: If specified, applies a causal mask as src_mask.
              Default: ``False``.
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ''
        if not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first :
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn._qkv_same_embed_dim :
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(src_mask, src_key_padding_mask, src)
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
                )


        x = src
        if not return_attn:
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
                x = x + self._ff_block(self.norm2(x))
            else:
                x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
                x = self.norm2(x + self._ff_block(x))
            return x
        else:
            if self.norm_first:
                x_sa, attn = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal, return_attn=return_attn)
                x = x + x_sa
                x = x + self._ff_block(self.norm2(x))
            else:
                x_sa, attn = self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal, return_attn=return_attn)
                x = self.norm1(x + x_sa)
                x = self.norm2(x + self._ff_block(x))
            return x, attn

    # self-attention block
    def _sa_block(self, x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor], is_causal: bool = False, return_attn: bool = False) -> torch.Tensor:
        x, attn = self.self_attn(x, x, x,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True, is_causal=is_causal, 
                                average_attn_weights=True)
        if return_attn:
            return self.dropout1(x), attn
        else:
            return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))