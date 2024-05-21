import torch
from fire import Fire
from tqdm import trange
from transformers import AutoModelForCausalLM


@torch.inference_mode()
def main(model_path, batch_size=1, seqlen_k=1, decode_k=1, attn="eager"):
    m = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation=attn,
    )

    unit = 1024**2  # MiB
    init_mem = torch.cuda.memory_reserved() / unit
    print(f"Model Weight Memory Usage: {init_mem:.0f} MiB")

    seqlen = int(seqlen_k * 1024)
    print(f"Sequence Length: {seqlen}")

    pkv = None
    input_ids = torch.zeros(batch_size, seqlen)
    input_ids = input_ids.to(torch.int64).cuda()

    oom_flag = False
    n_decode = decode_k * 1024
    with trange(n_decode, ncols=100) as prog:
        for i in prog:
            try:
                out = m(input_ids, past_key_values=pkv)
                pkv = out.past_key_values
                input_ids = torch.zeros(batch_size, 1)
                input_ids = input_ids.to(torch.int64).cuda()
            except:
                oom_flag = True
                break
            finally:
                total_mem = torch.cuda.memory_reserved() / unit
                delta_mem = total_mem - init_mem
                prog.desc = f"Memory: {delta_mem:.0f}/{total_mem:.0f} MiB"
                prog.desc += " (OOM)" if oom_flag else ""

    total_length = (seqlen + i) * batch_size
    print(f"Decode Length: {i}")
    print(f"Total Length: {total_length}")


if __name__ == "__main__":
    Fire(main)
