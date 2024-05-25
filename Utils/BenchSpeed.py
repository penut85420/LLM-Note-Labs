import random
import time
from concurrent.futures import ThreadPoolExecutor as TPE
from concurrent.futures import as_completed

from fire import Fire
from tqdm import tqdm


def main(n_prefill=1, n_decode=1, n_workers=1, n_requests=16, backend=None):
    prompts = prepare_data(n_prefill, n_requests)

    generate = dict(
        ggml=prepare_ggml_server,
        vllm=prepare_vllm_server,
        hf=prepare_hf_model,
    ).get(
        backend, prepare_server
    )(n_decode)

    n_prefill, n_decode = 0, 0
    delta = time.perf_counter()
    with tqdm(total=len(prompts), ncols=100) as prog:
        with TPE(max_workers=n_workers) as tpe:
            future = [tpe.submit(generate, prompt) for prompt in prompts]
            for resp in as_completed(future):
                n_inn, n_out = resp.result()
                n_prefill += n_inn
                n_decode += n_out

                ptps, dtps = summary(delta, n_prefill, n_decode)
                prog.desc = f"{ptps:.2f} / {dtps:.2f}"
                prog.update()


def prepare_data(inn_len, n):
    with open("wikitext2-test.txt", "rt", encoding="UTF-8") as fp:
        text = fp.read()
    return [rand_prompt(text, inn_len) for _ in range(n)]


def prepare_server(out_len):
    from text_generation import Client

    client = Client("http://localhost:8084/", timeout=600)

    def generate(prompt):
        resp = client.generate(prompt, max_new_tokens=out_len, decoder_input_details=True)
        return resp.details.prefill, resp.details.generated_tokens

    return generate


def prepare_ggml_server(out_len):
    import json

    import requests

    url = "http://127.0.0.1:8080/completion"

    def generate(prompt):
        data = dict(prompt=prompt, n_predict=out_len)
        resp = requests.post(url, json=data)
        resp = json.loads(resp.content)["timings"]
        return resp["prompt_n"], resp["predicted_n"]

    return generate


def prepare_vllm_server(out_len):
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="OuO")
    model_id = client.models.list().data[0].id

    def generate(prompt):
        resp = client.completions.create(model=model_id, prompt=prompt, max_tokens=out_len)
        return resp.usage.prompt_tokens, resp.usage.completion_tokens

    return generate


def prepare_hf_model(out_len):
    import os

    import torch
    from transformers import AutoModelForCausalLM as ModelImp
    from transformers import AutoTokenizer as TkImp
    from transformers import GenerationConfig
    from transformers import PreTrainedModel as ModelCls
    from transformers import PreTrainedTokenizer as TkCls

    model_path = os.getenv("HF_MODEL", None)
    attn_impl = os.getenv("HF_ATTN", "eager")

    tk: TkCls = TkImp.from_pretrained(
        model_path,
        padding_side="left",
        trust_remote_code=True,
    )

    m: ModelCls = ModelImp.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation=attn_impl,
    )

    gen_config = GenerationConfig(
        eos_token_id=tk.eos_token_id,
        pad_token_id=tk.eos_token_id,
        max_new_tokens=out_len,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate(prompt):
        inputs = tk(prompt, truncation=True, return_tensors="pt").to(device)
        output = m.generate(**inputs, generation_config=gen_config)
        prompt_len = len(inputs["input_ids"][0])
        decode_len = len(output[0]) - prompt_len
        return prompt_len, decode_len

    return generate


def summary(delta, n_prefill, n_decode):
    delta = time.perf_counter() - delta
    prefill_tps = n_prefill / delta
    decode_tps = n_decode / delta
    return prefill_tps, decode_tps


def rand_prompt(text, inn_len):
    rand = random.randint(0, len(text) - inn_len - 1)
    return text[rand : rand + inn_len]


if __name__ == "__main__":
    Fire(main)
