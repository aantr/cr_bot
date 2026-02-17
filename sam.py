from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model(
    bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    device="cuda",
    eval_mode=True,
    checkpoint_path="sam3.safetensors",
    load_from_HF=False,
)
processor = Sam3Processor(model, device="cuda")
state = processor.set_image("your_image.jpg")
state = processor.set_text_prompt("white bicycle", state)
print(state["masks"].shape)