import torch
from transformer import GPT
from load_weight import load_gpt2_weights
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

def top_k_filter(logits, k):
    # indices记录选出的logit在tensor中的位置
    values, indices = torch.topk(logits, k)
    # 生成和logits一样形状的张量，全填-inf
    filtered_logits = torch.full_like(logits, float('-inf'))
    # 把选出的logits填回去。这样softmax之后-inf一定不会被选中
    filtered_logits[indices] = values
    return filtered_logits


def generate(model, input_ids, max_new_tokens, temperature=0.8, top_k=50, eos_token_id=50256):
    model.eval()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)

        logits = logits[0, -1] / temperature
        logits = top_k_filter(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == eos_token_id:
            break
        # torch.cat要求张量维数相同
        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)

    return input_ids


if __name__ == "__main__":
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    my_model = GPT(vocab_size=50257, d_model=768, n_heads=12, n_layers=12, max_seq_len=1024)
    my_model = load_gpt2_weights(my_model, hf_model.state_dict())
    
    input_ids = torch.tensor([[464, 3797, 15070]])

    output = generate(my_model, input_ids, max_new_tokens=10)
    print(output)