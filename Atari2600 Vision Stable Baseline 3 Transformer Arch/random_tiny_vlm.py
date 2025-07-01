from architectures.CommandR.idefics2.modeling_idefics2_for_rl import *
from transformers import Idefics2Processor, LlavaProcessor, Idefics2ImageProcessor
from transformers import SiglipVisionModel, SiglipVisionConfig, MistralConfig
from PIL import Image

processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
).to(DEVICE)

# Create inputs
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What do we see in this image?"},
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "In this image, we can see the city of New York, and more specifically the Statue of Liberty."},
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "And how about this image?"},
        ]
    },
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image1, image2], return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}


# Generate
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts)
# ['User: What do we see in this image? \nAssistant: In this image, we can see the city of New York, and more specifically the Statue of Liberty. \nUser: And how about this image? \nAssistant: In this image we can see buildings, trees, lights, water and sky.']

quit()

# processor = Idefics2ImageProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", cache_dir="cache/models")
# processed_data = processor.preprocess(images=[Image.open("img.png"), Image.open("img_1.png")], return_tensors="pt")
processor2 = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir="cache/models")
processed_data2 = processor2(images=[Image.open("img.png"), Image.open("img_1.png")], text="<image>")

# print(processed_data)
# print(processed_data["pixel_values"].shape)
print(processed_data2["pixel_values"].shape)

config = Idefics2Config(
        use_cache=False,
        image_token_id=32_001,
        tie_word_embeddings=False,
        vision_config=Idefics2VisionConfig(
                                        hidden_size=32,
                                        intermediate_size=64,
                                        num_hidden_layers=2,
                                        num_attention_heads=2,
                                        num_channels=3,
                                        image_size=84,
                                        patch_size=8,
                                        hidden_act="gelu_pytorch_tanh",
                                        layer_norm_eps=1e-6,
                                        attention_dropout=0.0,
                                        initializer_range=0.02,
                                        ),
        perceiver_config=None,
        text_config=MistralConfig(
                                vocab_size=32000,
                                hidden_size=64,
                                intermediate_size=128,
                                num_hidden_layers=2,
                                num_attention_heads=4,
                                num_key_value_heads=1,
                                hidden_act="silu",
                                max_position_embeddings=4096 * 32,
                                initializer_range=0.02,
                                rms_norm_eps=1e-6,
                                use_cache=True,
                                pad_token_id=None,
                                bos_token_id=1,
                                eos_token_id=2,
                                tie_word_embeddings=False,
                                rope_theta=10000.0,
                                sliding_window=4096,
                                attention_dropout=0.0,
                                )
        )
print(config)
print(config.vision_config)

model = Idefics2ForConditionalGeneration(config)

print(processed_data2)
model.generate(*processed_data2)

print(model)
