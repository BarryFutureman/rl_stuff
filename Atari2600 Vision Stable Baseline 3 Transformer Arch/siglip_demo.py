from transformers.models.siglip.modeling_siglip import SiglipVisionModel, SiglipVisionConfig
from transformers import SiglipImageProcessor
from PIL import Image

model = SiglipVisionModel(SiglipVisionConfig(
                            hidden_size=128,
                            intermediate_size=256,
                            num_hidden_layers=4,
                            num_attention_heads=4,
                            num_channels=3,
                            image_size=84,
                            patch_size=16,
                            hidden_act="gelu_pytorch_tanh",
                            layer_norm_eps=1e-6,
                            attention_dropout=0.0,)
                        )

print(model)

processor = SiglipImageProcessor(
        do_resize=True,
        size=(84, 84),
        do_rescale=True
)

print(processor)

img = Image.open("img.png")
inputs = processor.preprocess(img, return_tensors="pt")
print(inputs["pixel_values"].shape)

output = model(pixel_values=inputs["pixel_values"])

print(output.last_hidden_state.shape)
