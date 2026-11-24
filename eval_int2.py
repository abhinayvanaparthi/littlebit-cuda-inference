import time
import torch
from run_qwen3_int2 import load_int2_model

model, tokenizer = load_int2_model("./qwen3-8b-int2-stage3")
model.eval()

questions = [
    "Explain the difference between a CPU and a GPU in 3 short bullet points.",
    "Write a Python function to check if a word is a palindrome, and give one example.",
    "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Explain briefly."
]

stop_ids = {tokenizer.eos_token_id, 151643, 151645}

def generate_reply(prompt_text, max_new_tokens=160, temperature=0.7, top_p=0.9, rep_penalty=1.15):
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to("cuda:0")
    generated = input_ids[0].tolist()
    t0 = time.perf_counter()

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]

        for _ in range(max_new_tokens):
            for tok_id in set(generated):
                if logits[0, tok_id] > 0:
                    logits[0, tok_id] /= rep_penalty
                else:
                    logits[0, tok_id] *= rep_penalty

            probs = torch.softmax(logits / temperature, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cum_probs - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            next_tok = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))

            tok_id = next_tok.item()
            if tok_id in stop_ids:
                break
            generated.append(tok_id)
            print(tokenizer.decode([tok_id]), end="", flush=True)

            out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]

    elapsed = time.perf_counter() - t0
    new_tokens = len(generated) - input_ids.shape[1]
    return new_tokens, elapsed

for i, q in enumerate(questions, 1):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    print(f"\n[{i}] Q: {q}\n    A: ", end="", flush=True)
    n_tok, dt = generate_reply(prompt)
    print(f"\n    ({n_tok} tokens in {dt:.1f}s -> {n_tok / max(dt, 1e-6):.2f} tok/s)\n" + "-" * 70)

print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB / 4.00 GB")
