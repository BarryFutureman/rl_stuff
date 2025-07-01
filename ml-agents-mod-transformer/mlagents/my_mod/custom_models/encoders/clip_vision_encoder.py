import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from mlagents.my_mod.custom_models.my_transformer_implementation import EmbeddingOnlyTransformerBody


class CLIPImageEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.select_layer = -2
        self.clip_vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16",
                                                                 cache_dir="cache/models")
        self.clip_vision_model.requires_grad_(False)

        self.transformer_body = EmbeddingOnlyTransformerBody(input_t=196, input_n_embed=768, num_layers=2,
                                                             num_heads=8, hidden_size=hidden_size)

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        image_features = image_features[:, 1:]

        return image_features

    @torch.no_grad()
    def forward(self, images):
        image_forward_outs = self.clip_vision_model(images.to(device=self.device, dtype=self.dtype),
                                                    output_hidden_states=True)

        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        # print("<CLIPVisionTower>image_feature", image_features.size())

        # Forward image feature to transformer body
        image_feature_vectors = self.transformer_body(image_features)

        return image_feature_vectors

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.clip_vision_model.dtype

    @property
    def device(self):
        return self.clip_vision_model.device

    @property
    def config(self):
        if self.is_loaded:
            return self.clip_vision_model.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
