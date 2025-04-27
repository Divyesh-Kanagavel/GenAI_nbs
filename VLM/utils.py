from modeling_gemma import PaliGemmaforConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os

# load the paligemma model from hf
def load_hf_model(
        model_path, device
):       
    tokenizer = AutoTokenizer.from_pretrained(model_path,padding_side="right")
    assert tokenizer.padding_side == "right"
    safetensors_path = glob.glob(os.path.join(model_path, "*.safetensors"))

    tensors= {}
    for safetensors_file in safetensors_path:
        with safe_open(safetensors_file, framework="pt",device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    with open(os.path.join(model_path, "config.json"),"r") as f:
        config_file = json.load(f)
        config = PaliGemmaConfig(**config_file)
    
    model = PaliGemmaforConditionalGeneration(config).to(device)

    model.load_state_dict(tensors, strict=False)
    
    model.tie_weights()

    return model , tokenizer





    
