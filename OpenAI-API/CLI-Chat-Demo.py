import tiktoken
from openai import OpenAI


def main():
    client = OpenAI()
    tk = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def truncate(messages, limit=300):
        """
        我們從訊息的尾端往前拜訪，不斷累加總 Token 數
        直到總 Token 數超過限制，最後將 System Prompt 加回訊息裡面
        """
        total = 0
        new_messages = list()
        for msg in reversed(messages[1:]):
            total += len(tk.encode(msg["content"]))
            if total > limit:
                break
            new_messages.insert(0, msg)
        new_messages.insert(0, messages[0])
        return new_messages

    def chat(messages):
        return client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True,
        )

    messages = [{"role": "system", "content": "你現在是一個使用繁體中文的貓娘。"}]

    while True:
        prompt = input("User: ").strip()
        messages.append({"role": "user", "content": prompt})

        messages = truncate(messages)
        response = chat(messages)

        print(end="Assistant: ", flush=True)
        full_resp = str()
        for resp in response:
            if not resp.choices:
                continue

            token = resp.choices[0].delta.content
            if token:
                print(end=token, flush=True)
                full_resp += token
        print()

        # 將模型輸出加入歷史訊息
        messages.append({"role": "assistant", "content": full_resp})


if __name__ == "__main__":
    main()
