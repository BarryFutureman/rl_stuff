from typing import Callable, List, Dict, Tuple, Optional, Union, Any

from mlagents.torch_utils import torch, nn, default_device
from enum import Enum

from mlagents_envs.base_env import ObservationSpec, ObservationType
from mlagents.trainers.settings import NetworkSettings, EncoderType
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.torch_entities.layers import Swish
from mlagents.trainers.torch_entities.encoders import VectorInput
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.trajectory import ObsUtil
import math
from dataclasses import dataclass
from torch.nn import functional as F
from mlagents.my_mod.custom_models.encoders.clip_vision_encoder import CLIPImageEncoder
from mlagents.my_mod.custom_models.encoders.vector_processor import VectorInput
from mlagents.trainers.torch_entities.encoders import NatureVisualEncoder


# TODO: USE ADAM-Warmup instead


class CLIPBasedObservationEncoder(nn.Module):
    ATTENTION_EMBEDDING_SIZE = 128  # The embedding size of attention is fixed

    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        h_size: int,
        vis_encode_type: EncoderType,
        normalize: bool = False,
    ):
        """
        Returns an ObservationEncoder that can process and encode a set of observations.
        Will use an RSA if needed for variable length observations.
        """
        super().__init__()
        self.processors = []
        self.embedding_sizes = []
        for obs_spec in observation_specs:
            obs_spec_shape = obs_spec.shape
            obs_spec_dim_prop = obs_spec.dimension_property
            if obs_spec_dim_prop in ModelUtils.VALID_VISUAL_PROP:
                visual_encoder_class = ModelUtils.get_encoder_for_type(vis_encode_type)
                vision_encoder_h_size = h_size//2*3
                new_processor = visual_encoder_class(obs_spec_shape[1], obs_spec_shape[2], obs_spec_shape[0], vision_encoder_h_size) # CLIPImageEncoder(hidden_size=vision_encoder_h_size)
                new_embed_size = vision_encoder_h_size
                self.processors.append(new_processor)
                self.embedding_sizes.append(new_embed_size)
            elif obs_spec_dim_prop in ModelUtils.VALID_VECTOR_PROP:
                self.processors.append(VectorInput(obs_spec_shape[0], normalize))
                self.embedding_sizes.append(obs_spec_shape[0])
            else:
                print("No Processor Suitable!")
                quit()

        self.normalize = normalize
        self._total_enc_size = sum(self.embedding_sizes)

        self._total_goal_enc_size = 0
        self._goal_processor_indices: List[int] = []

    @property
    def total_enc_size(self) -> int:
        """
        Returns the total encoding size for this ObservationEncoder.
        """
        return self._total_enc_size

    @property
    def total_goal_enc_size(self) -> int:
        """
        Returns the total goal encoding size for this ObservationEncoder.
        """
        return self._total_goal_enc_size

    def update_normalization(self, buffer: AgentBuffer) -> None:
        obs = ObsUtil.from_buffer(buffer, len(self.processors))
        for vec_input, enc in zip(obs, self.processors):
            if isinstance(enc, VectorInput):
                enc.update_normalization(torch.as_tensor(vec_input.to_ndarray()))

    def copy_normalization(self, other_encoder: "CLIPBasedObservationEncoder") -> None:
        if self.normalize:
            for n1, n2 in zip(self.processors, other_encoder.processors):
                if isinstance(n1, VectorInput) and isinstance(n2, VectorInput):
                    n1.copy_normalization(n2)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Encode observations.
        :param inputs: List of Tensors corresponding to a set of obs.
        """
        encodes = []

        for index, processor in enumerate(self.processors):
            obs_input = inputs[index]
            processed_obs = processor(obs_input)
            encodes.append(processed_obs)

        encoded_self = torch.cat(encodes, dim=1)

        return encoded_self


class VisionTransformerNetworkBody(nn.Module):
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        encoded_act_size: int = 0,
    ):
        super().__init__()
        self.normalize = network_settings.normalize
        self.use_lstm = network_settings.memory is not None
        self.h_size = network_settings.hidden_units
        self.m_size = (
            network_settings.memory.memory_size
            if network_settings.memory is not None
            else 0
        )
        self.observation_encoder = CLIPBasedObservationEncoder(
            observation_specs,
            self.h_size,
            EncoderType.NATURE_CNN,
            self.normalize,
        )
        # self.observation_encoder = NatureVisualEncoder(
        #     width=84, height=84, initial_channels=3, output_size=self.h_size
        # )
        self.processors = self.observation_encoder.processors

        self.obs_context_len = 2
        # self.obs_context = torch.zeros((1, self., self.h_size))

        self._body_encoder = TransformerBody(block_size=self.obs_context_len,
                                             input_vector_size=self.observation_encoder.total_enc_size//self.obs_context_len,
                                             num_layers=network_settings.num_layers,
                                             num_heads=2,
                                             hidden_size=self.h_size
                                             )

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.observation_encoder.update_normalization(buffer)

    def copy_normalization(self, other_network: "VisionTransformerNetworkBody") -> None:
        self.observation_encoder.copy_normalization(other_network.observation_encoder)

    @property
    def memory_size(self) -> int:
        return self.m_size

    def forward(
        self,
        inputs: List[torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """encoded_time_steps = []
        for time_step in inputs:
            encoded_self = self.observation_encoder(time_step)
            # Add time step dimension
            encoded_self = encoded_self.unsqueeze(1)
            encoded_time_steps.append(encoded_self)
        encoded_time_steps = torch.cat(encoded_time_steps, 1)"""

        # Input shape is [past_context, new_obs]
        encoded_self = self.observation_encoder(inputs)

        # print("memories:", memories, memories.size())
        encoding, new_context = self._body_encoder(encoded_self, memories)

        return encoding, new_context


class Initialization(Enum):
    Zero = 0
    XavierGlorotNormal = 1
    XavierGlorotUniform = 2
    KaimingHeNormal = 3  # also known as Variance scaling
    KaimingHeUniform = 4
    Normal = 5


_init_methods = {
    Initialization.Zero: torch.zero_,
    Initialization.XavierGlorotNormal: torch.nn.init.xavier_normal_,
    Initialization.XavierGlorotUniform: torch.nn.init.xavier_uniform_,
    Initialization.KaimingHeNormal: torch.nn.init.kaiming_normal_,
    Initialization.KaimingHeUniform: torch.nn.init.kaiming_uniform_,
    Initialization.Normal: torch.nn.init.normal_,
}


def linear_layer(
    input_size: int,
    output_size: int,
    kernel_init: Initialization = Initialization.XavierGlorotUniform,
    kernel_gain: float = 1.0,
    bias_init: Initialization = Initialization.Zero,
) -> torch.nn.Module:
    """
    Creates a torch.nn.Linear module and initializes its weights.
    :param input_size: The size of the input tensor
    :param output_size: The size of the output tensor
    :param kernel_init: The Initialization to use for the weights of the layer
    :param kernel_gain: The multiplier for the weights of the kernel. Note that in
    TensorFlow, the gain is square-rooted. Therefore calling  with scale 0.01 is equivalent to calling
        KaimingHeNormal with kernel_gain of 0.1
    :param bias_init: The Initialization to use for the weights of the bias layer
    """
    layer = torch.nn.Linear(input_size, output_size)
    if (
        kernel_init == Initialization.KaimingHeNormal
        or kernel_init == Initialization.KaimingHeUniform
    ):
        _init_methods[kernel_init](layer.weight.data, nonlinearity="linear")
    else:
        _init_methods[kernel_init](layer.weight.data)
    layer.weight.data *= kernel_gain
    _init_methods[bias_init](layer.bias.data)
    return layer


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                             dropout_p=0,
                                                             is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class TransformerConfig:
    block_size: int = 12
    vector_size: int = 6
    hidden_size: int = 256
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False


class TransformerBody(nn.Module):
    def __init__(self,
                 block_size: int,
                 input_vector_size: int,
                 num_layers: int,
                 num_heads: int,
                 hidden_size: int,
                 kernel_init: Initialization = Initialization.KaimingHeNormal,
                 kernel_gain: float = 1.0, ):
        super().__init__()

        model_args = dict(n_layer=num_layers, n_head=num_heads, n_embd=hidden_size, block_size=block_size,
                          vector_size=input_vector_size, bias=False, hidden_size=hidden_size, dropout=0)
        config = TransformerConfig(**model_args)
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=linear_layer(
                config.vector_size, config.n_embd,
                kernel_init=kernel_init,
                kernel_gain=kernel_gain,
            ),
            swish=Swish(),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.transformer_head = linear_layer(
            config.n_embd,
            config.hidden_size,
            kernel_init=kernel_init,
            kernel_gain=kernel_gain,
        )
        self.pos = torch.arange(0, config.block_size, dtype=torch.long, device=default_device())

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params(non_embedding=False) / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_vectors, memory_embeddings):
        # Turn input into embeddings:
        # Unsqueeze to add time step dimension
        # input_vectors = input_vectors.unsqueeze(1)
        input_vectors = input_vectors.view(input_vectors.size()[0], self.config.block_size, self.config.vector_size)
        input_embeddings = self.transformer.wte(input_vectors)

        # Trim the memory for adding new memory
        # TODO: Maybe there is a better way?
        # memory_embeddings = memory_embeddings[:, 1:, :]

        # Combine current input and memory
        # new_memory = torch.cat([memory_embeddings, input_embeddings], 1)
        pos_emb = self.transformer.wpe(self.pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(input_embeddings + pos_emb)

        """# Turn input into embeddings:
        input_embeddings = self.transformer.wte(input_vectors)
        pos_emb = self.transformer.wpe(self.pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(input_embeddings + pos_emb)"""

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.transformer_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
        # Squeeze because we only want the last dim
        logits = logits.squeeze()
        logits = self.transformer.swish(logits)

        return logits, None  # new_memory
