"""Convert the HauhauCS Qwen3.5 4B safetensors reconstruction for vLLM."""

import os

from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


MODEL_ID = os.environ.get(
    "SOURCE_MODEL",
    "DreamFast/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Safetensor-Benchmark",
)
SAVE_DIR = os.environ.get("SAVE_DIR", "/output")


model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="cpu",
    dtype="auto",
    low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Match the supported Qwen3.5 W4A16 layout: the visual encoder, language-model
# head, and Gated DeltaNet linear-attention projections remain in BF16.
recipe = QuantizationModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "lm_head",
        "re:.*visual.*",
        "re:.*linear_attn.*",
    ],
)

oneshot(model=model, recipe=recipe)
model.save_pretrained(SAVE_DIR, save_compressed=True, safe_serialization=True)
processor.save_pretrained(SAVE_DIR)
