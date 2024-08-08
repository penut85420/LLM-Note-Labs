import torch
from fire import Fire
from tqdm import trange
from transformers import AutoModelForCausalLM


@torch.inference_mode()
def main(
    model_path,  # 模型路徑
    batch_size=1,  # 同時推論筆數
    k_prefill=1,  # 輸入長度，以 1024 為單位
    k_decode=1,  # 輸出長度，以 1024 為單位
    attn="eager",  # 注意力實做種類，可為 eager, sdpa, flash_attention_2
):
    # 讀取模型
    m = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation=attn,
    )

    # 取得讀取模型後的 GPU 記憶體使用量
    unit = 1024**2  # MiB
    init_mem = torch.cuda.memory_reserved() / unit
    print(f"Model Weight Memory Usage: {init_mem:.0f} MiB")

    # 計算輸入長度
    n_prefill = int(k_prefill * 1024)
    print(f"Sequence Length: {n_prefill}")

    # pkv (past_key_values) 即為 KV Cache
    pkv = None

    # 製作假的 Prefilling 輸入
    input_ids = torch.zeros(batch_size, n_prefill)
    input_ids = input_ids.to(torch.int64).cuda()

    # 製作假的 Decoding 輸入
    input_ids_1 = torch.zeros(batch_size, 1)
    input_ids_1 = input_ids_1.to(torch.int64).cuda()

    # 開始進行 Decoding
    oom_flag = False
    n_decode = int(k_decode * 1024)
    with trange(n_decode, ncols=100) as prog:
        for i in prog:
            try:
                out = m(input_ids, past_key_values=pkv)
                pkv = out.past_key_values
                input_ids = input_ids_1
            except Exception:
                i -= 1
                oom_flag = True
                break
            finally:
                # 在 Decoding 的過程中顯示 GPU 記憶體使用量的變化
                total_mem = torch.cuda.memory_reserved() / unit
                delta_mem = total_mem - init_mem
                prog.desc = f"Memory: {delta_mem:.0f}/{total_mem:.0f} MiB"
                prog.desc += " (OOM)" if oom_flag else ""

    i += 1
    total_length = (n_prefill + i) * batch_size
    print(f"Decode Length: {i}")
    print(f"Total Length: {total_length}")


if __name__ == "__main__":
    Fire(main)
