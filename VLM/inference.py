from PIL import Image
import fire 
import torch
import torch.nn.functional as F

from process_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaforConditionalGeneration
from utils import load_hf_model

def _sample_top_p(next_token_logits, top_p):
    sorted_probs, sorted_indices = torch.sort(next_token_logits, dims = -1, descending=True)
    probs_sum = torch.cumsum(sorted_probs,dim=-1)

    mask = probs_sum - sorted_probs > top_p
    sorted_probs[mask] = 0.0 # zero out the probs that are above top_p by sum

    sorted_probs.div_(sorted_probs.sum(dim=-1,keepdim=True))

    next_token = torch.multinomial(sorted_probs, num_samples=1)

    next_token = torch.gather(sorted_indices,-1, next_token)
    return next_token


def move_inputs_to_device(model_inputs, device):
    model_inputs= {k:v.to(device) for k,v in model_inputs.items()}
    return model_inputs

def get_model_inputs(processor, prompt, image_file_path, device):
    image = Image.open(image_file_path)
    images =[image] # we use a single image,but multi images can be used
    prompts = [prompt] # same idea as above
    model_inputs = processor(images, prompts)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs

def test_inference(
        model, processor, device,
        prompt, image_file_path, max_tokens_to_generate,
        temperature, top_p, do_sample
):
    model_inputs= get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()


    # eos token
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    for _ in range(max_tokens_to_generate):
        outputs = model(
            input_ids,
            pixel_values,
            attention_mask,
            kv_cache
        )
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :] # select the last token in the sequence

        if do_sample:
            next_token_logits = F.softmax(next_token_logits/temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1,keepdim=True)
        
        assert next_token.size() == (1,1)

        next_token = next_token.squeeze(0)
        generated_tokens.append(next_token)

        if next_token.item() == stop_token:
            break

        input_ids = next_token.unsqueeze(0)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1,1), device=input_ids.device)],dim=-1
        )

    generated_tokens = torch.cat(generated_tokens, dim=-1)
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(prompt + decoded)




def main(
        model_path = None,
        prompt = None,
        image_file_path = None,
        max_tokens_to_generate = 512,
        temperature= 0.0,
        top_p = 0.0,
        do_sample = False,
        only_cpu = False

):
    device="cpu"
    if not only_cpu:
        device = "mps"
    print("Device : ", device)
    print("Loading model ... ")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens,image_size)

    print("Running inference...")
    with torch.no_grad():
        test_inference(
            model,processor,device,
            prompt,image_file_path, max_tokens_to_generate,
            temperature, top_p,do_sample,
        )


if __name__ == "__main__":
    fire.Fire(main)