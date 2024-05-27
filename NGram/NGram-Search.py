import json
import os
from collections import defaultdict

from fire import Fire


def ngram_search(query="語言模型", target_dir="."):
    # 建立 N-Gram 索引
    index = defaultdict(set)
    chunks = list()
    for full_path in iter_markdown(target_dir):
        with open(full_path, "rt", encoding="UTF-8") as fp:
            # 以雙換行作為段落邊界
            segments = fp.read().split("\n\n")

        for chunk in segments:
            # 將所有 N-Gram 都當成索引
            # 並將 Chunk ID 放進索引對應的列表
            for seg in all_ngram(chunk, 1, 6):
                index[seg].add(len(chunks))
            chunks.append(chunk)

    # 將索引與區塊存下來
    with open("index.json", "wt", encoding="UTF-8") as fp:
        _index = {k: list(index[k]) for k in index}
        json.dump(_index, fp, ensure_ascii=False)

    with open("chunks.json", "wt", encoding="UTF-8") as fp:
        json.dump(chunks, fp, ensure_ascii=False)

    # 開始實際查詢
    q_len = len(query)
    query_ngrams = {s for s in all_ngram(query, 1, len(query))}

    results = []
    for i in index[query]:
        # 計算 Jaccard Similarity 當作排名依據
        score = calc_jaccard(query_ngrams, q_len, chunks[i])
        results.append((score, chunks[i]))
    results = sorted(results, reverse=True)

    # 輸出前五名的結果
    for i, (score, res) in zip(range(5), results):
        print(f"Rank {i}, Score: {score:.4f}, Chunk: {repr(res)}")


def iter_markdown(target_dir):
    # 拜訪所有 Markdown 文件
    for dir_path, _, file_list in os.walk(target_dir):
        for file_name in file_list:
            if not file_name.endswith(".md"):
                continue
            yield os.path.join(dir_path, file_name)


def ngram(text: str, n: int):
    # 取得指定大小的 N-Gram
    for i in range(0, len(text) - n + 1):
        yield text[i : i + n]


def all_ngram(text: str, a, b, step=1):
    # 取得範圍內所有大小的 N-Gram
    a = max(1, a)
    b = min(len(text), b)
    for i in range(a, b + 1, step):
        for seg in ngram(text, i):
            yield seg


def calc_jaccard(query_ngrams, q_len, chunk: str):
    # 計算 Jaccard Similarity
    chunk_ngrams = {seg for seg in all_ngram(chunk, 1, q_len)}

    inter = query_ngrams & chunk_ngrams
    union = query_ngrams | chunk_ngrams

    return len(inter) / len(union)


if __name__ == "__main__":
    Fire(ngram_search)
