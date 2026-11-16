"""
加载gpt2-small权重
"""
from transformers import GPT2LMHeadModel
from transformer import GPT
import torch

"""
由用户创建的tensor默认都是需要记录梯度的叶子节点比如llm的各种参数
对它们进行原地修改会被当成一次计算记录在PyTorch的有向计算图里
PyTorch也是根据这个来记录计算得到的中间tensor来源于哪个叶子节点
以此进行梯度的反向传播
因此这里加载参数需要加上no_grad修饰符
"""
@torch.no_grad()
def load_gpt2_weights(my_model, hf_state_dict, n_layer=12):
    sd = hf_state_dict
    d_model = my_model.token_emb.embedding_dim  # 768

    # 用 copy_ 而不是 .data = ...：形状不一致时会直接报错，不会悄悄换成错误形状的张量
    def copy(param, tensor):
        assert param.shape == tensor.shape, f"形状不匹配: {tuple(param.shape)} vs {tuple(tensor.shape)}"
        # 带_的都是原地修改方法，也就是把tensor的值原地复制进param，param对象不变
        param.copy_(tensor)

    # embedding 层,原样赋值
    # 输出头和 token_emb 共享权重（见 GPT.__init__），所以这里不用单独加载 head
    copy(my_model.token_emb.weight, sd["transformer.wte.weight"])
    copy(my_model.pos_emb.weight, sd["transformer.wpe.weight"])

    for i in range(n_layer):
        block = my_model.blocks[i]
        prefix = f"transformer.h.{i}."

        # LayerNorm:原样赋值
        copy(block.ln1.weight, sd[prefix + "ln_1.weight"])
        copy(block.ln1.bias,   sd[prefix + "ln_1.bias"])
        copy(block.ln2.weight, sd[prefix + "ln_2.weight"])
        copy(block.ln2.bias,   sd[prefix + "ln_2.bias"])

        # Attention:
        # GPT-2 用的是 Conv1D，权重形状是 (in, out)，nn.Linear 是 (out, in)，所以要转置
        # c_attn 把 Q、K、V 拼在一起：(768, 2304) = 3 个 (768, 768)，要沿最后一维切成三份
        q_w, k_w, v_w = sd[prefix + "attn.c_attn.weight"].split(d_model, dim=1)
        q_b, k_b, v_b = sd[prefix + "attn.c_attn.bias"].split(d_model, dim=0)
        copy(block.attn.W_q.weight, q_w.T)
        copy(block.attn.W_k.weight, k_w.T)
        copy(block.attn.W_v.weight, v_w.T)
        copy(block.attn.W_q.bias, q_b)
        copy(block.attn.W_k.bias, k_b)
        copy(block.attn.W_v.bias, v_b)
        copy(block.attn.W_o.weight, sd[prefix + "attn.c_proj.weight"].T)
        copy(block.attn.W_o.bias,   sd[prefix + "attn.c_proj.bias"])

        # FFN:c_fc / c_proj 都要转置
        copy(block.ffn.W1.weight, sd[prefix + "mlp.c_fc.weight"].T)
        copy(block.ffn.W1.bias,   sd[prefix + "mlp.c_fc.bias"])
        copy(block.ffn.W2.weight, sd[prefix + "mlp.c_proj.weight"].T)
        copy(block.ffn.W2.bias,   sd[prefix + "mlp.c_proj.bias"])

    # 最终 LayerNorm
    copy(my_model.ln_f.weight, sd["transformer.ln_f.weight"])
    copy(my_model.ln_f.bias,   sd["transformer.ln_f.bias"])

    return my_model


if __name__ == "__main__":
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_model.eval() # eval方法把模型切换为推理模式，对于训练和推理时行为不同的层有用

    my_model = GPT(vocab_size=50257, d_model=768, n_heads=12, n_layers=12, max_seq_len=1024)
    my_model = load_gpt2_weights(my_model, hf_model.state_dict())
    my_model.eval()

    input_ids = torch.tensor([[464, 3797, 15070]])
    with torch.no_grad(): # 推理时开着节省生成计算图的资源
        logits = my_model(input_ids) # （batch, seq_len, d_model）
        hf_logits = hf_model(input_ids).logits

    print(logits.shape)
    # 和 Hugging Face 官方实现对比：最大误差应该在 1e-4 量级以内
    print("与官方实现的最大误差:", (logits - hf_logits).abs().max().item())
    next_token_id = logits[0, -1].argmax().item() # 最后一词的最大logit
    print("预测的下一个 token:", next_token_id, "| 官方:", hf_logits[0, -1].argmax().item())
