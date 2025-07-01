from torch import nn
from typing import Callable, Dict, List, Sequence, Tuple, Type, Union
import torch
from pi0_dit.modeling_pi0 import PI0FlowMatching, DiTConfig


class TransformerModule(nn.Module):
    def __init__(
            self,
            in_features: int | None = None,
            out_features: int | torch.Size = None,
            device: str | None = None,
            load_path: str | None = None,
            hidden_size: int = 32,
            intermediate_size: int = 192,
            num_layers: int = 4,
            context_length: int = 4,
    ):
        super().__init__()
        self.device = device
        if load_path:
            self.load_model_weights(load_path)
        else:
            print("\033[36m[Initializing model]\033[39m")
            if isinstance(out_features, torch.Size):
                out_features = out_features[-1]

            action_chunk_size = 4
            self.config = DiTConfig(
                obs_dim=int(in_features),
                max_action_dim=int(out_features) // action_chunk_size,
                proj_width=64,
                n_action_steps=action_chunk_size,
                num_steps=4,
                use_cache=True,
            )
            self.model = PI0FlowMatching(
                self.config
            ).to(self.device)

    def load_model_weights(self, load_path):
        print(f"\033[36m[Loading model from {load_path}]\033[39m")
        print(PI0FlowMatching)
        self.model = PI0FlowMatching.from_pretrained(load_path).to(self.device)
        self.config = self.model.config

        # Ensure all parameters are set to require gradients
        for param in self.model.parameters():
            param.requires_grad = True

        # Set model to training mode
        self.model.train()

    def forward(self, *inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        if len(inputs) > 1:
            raise NotImplementedError()

        inputs_vectors = inputs[0]

        *batch, T, D = inputs_vectors.shape

        if len(batch) > 1:
            inputs_vectors = inputs_vectors.flatten(0, len(batch) - 1)

        logits = self.model(
            inputs_embeds=inputs_vectors,
        )
        logits = logits.flatten(start_dim=-2)

        if len(batch) > 1:
            logits = logits.unflatten(0, batch)

        return logits


if __name__ == '__main__':
    # t_m = TransformerModule(in_features=30, out_features=40, hidden_size=128, num_layers=18, context_length=1024)
    # t_m = TransformerModule(in_features=30, out_features=40, hidden_size=96, num_layers=8, context_length=1024)
    t_m = TransformerModule(in_features=30, out_features=40, hidden_size=64, intermediate_size=192, num_layers=8,
                            context_length=1024)
    print(t_m)
    print(f"{t_m.model.num_parameters() / 1e6:.2f} Million Parameters")
