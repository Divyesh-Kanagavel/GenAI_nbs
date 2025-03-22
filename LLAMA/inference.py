from typing import Optional
import torch
from sentencepiece import SentencePieceProcessor
import time
from tqdm import tqdm
from pathlib import Path
import json

from model import Transformer, Model_args

class LLAMA:
    def __init__(self, model : Transformer, tokenizer : SentencePieceProcessor, model_args : Model_args):
        self.args = model_args
        self.model = model
        self.tokenizer = tokenizer
    
    @staticmethod
    def build(checkpoints_dir : str, tokenizer_path : str, load_model : bool, max_seq_len : int, max_batch_size : int, device : str):
        # load the model checkpoint from the .pth file downloaded from llama website
        print("Loading model...")
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0 , "No checkpoints found"
            chk_path = checkpoints[0]
            print(f'Loading checkpoint from {chk_path}')
            checkpoint = torch.load(chk_path, map_location="cpu")
            print(f"loaded checkpoint in {(time.time()-prev_time):.2f} secomds")
            prev_time = time.time()
        
        with open(Path(checkpoints_dir) / 'params.json','r') as f:
            params = json.loads(f.read())
        model_args : Model_args = Model_args(
            max_seq_len = max_seq_len,
            max_batch_size = max_batch_size,
            device = device,
            # read other params from params.json file
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        # set the vocab size from the tokenizer
        model_args.vocab_size = tokenizer.vocab_size()

        # set the datatype of the tensors
        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor) # use half precision
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor) # use bfloat16 precision
        
        # create the model
        model  = Transformer(model_args).to(device)
        #load model paramters from the checkpoint
        if load_model:
            del checkpoint["rope.freqs"]
            # if strict is True, it will raise an error if the names used in our class differ from checkpoint keys
            model.load_state_dict(checkpoint, strict=True)
            print(f"loaded model in {(time.time()-prev_time):.2f} seconds")
        return LLAMA(model, tokenizer, model_args)
    
    def _sample_top_p(self, probs : torch.Tensor, top_p : float):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0

        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True)) # normalization into probabilities after cumsum and shift

        next_token = torch.multinomial(probs_sort, num_samples = 1)
        next_token = torch.gather(probs_idx,-1,next_token)
        return next_token
    
    def text_completion(self,prompts : list[str], temperature : float = 0.6, top_p : float = 0.9, max_gen_len : Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos = True, add_eos = False) for prompt in prompts]
        batch_size = len(prompts) # number of prompts

        assert batch_size <= self.args.max_batch_size

        # make sure the prompt length is not greater than max_seq_len
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.args.max_seq_len

        total_len = min(self.args.max_seq_len, max_prompt_len + max_gen_len)

        # create the list which will contain the generated tokens + prompt tokens
        # pad all the tokens initially
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        # populate the prompt tokens
        for k, t in enumerate(prompt_tokens):
            tokens[k,:len(t)] = torch.tensor(t, dtype=torch.long,device=device)
        
        # is eos reached
        eos_reached = torch.tensor([False]*batch_size, device=device)
        # generate mask for the tokens
        prompt_mask_tokens = tokens!=pad_id

        # generate tokens for len total_len
        for cur_pos in tqdm(range(1, total_len),desc="Generating tokens!"):
            with torch.no_grad():
                logits = self.model.forward(tokens[:,cur_pos-1:cur_pos],cur_pos)
            # if temperature is available, use top p sampling i.e select those tokens which have cumulative sum of given probability
            if temperature > 0.0:
                probs = torch.softmax(logits[:,-1]/ temperature,dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            # Greedy sampling, select the token with highest probability
            else:
                next_token = torch.argmax(logits[:,-1],dim=-1)
            
            next_token = next_token.reshape(-1)
            # don't populate the next token into the sequence if there is a prompt token already
            next_token = torch.where(prompt_mask_tokens[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            # check if eos is reached
            eos_reached |= (~prompt_mask_tokens[:, cur_pos]) & (next_token == self.tokenizer.eos_id())

            if all(eos_reached):
                break
        # decoding tokens into text
        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # check if eos token is present
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                # select prompt tokens till eos index
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            
            out_tokens.append(current_prompt_tokens)
            # decode text from the output tokens
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return (out_tokens, out_text)

if __name__ == "__main__":
    torch.manual_seed(0)
    allow_cuda = False
    # set the device to cpu
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"
    prompts = [
        "The Bernoulli's principle states that ",
        "The importance of having many trees is that ",
        "How to spot a human among aliens ? is it even possible?",
        """What is the pattern here? 
            a - 1
            c - 3
            x - 25
            y - ?
        """
    ]
    model = LLAMA.build(
        checkpoints_dir = "/Users/divyeshkanagavel/.llama/checkpoints/Llama-2-7b/",
        tokenizer_path = "/Users/divyeshkanagavel/.llama/checkpoints/Llama-2-7b/tokenizer.model",
        load_model = True,
        max_seq_len = 1024,
        max_batch_size = len(prompts),
        device = device
    )

    print("model loaded successfully!!")

    # calling the model to generate tokens for the given prompts
    out_tokens, out_text  = model.text_completion(prompts, max_gen_len = 64)
    assert len(out_text) == len(prompts) # batch size is preserved

    for i in range(len(out_text)):
        print(f'{out_text[i]}')
        print('-'*50) # separator between text for each prompt






        


        