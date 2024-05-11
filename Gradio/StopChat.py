import random
import time
from threading import Event

import gradio as gr


def get_resp():
    r1 = random.randint(10, 20)  # "喵" 的數量
    r2 = random.randint(1, 3)  # "！" 的數量
    resp = "喵" * r1 + "！" * r2

    for ch in resp:
        time.sleep(0.3)  # 模擬文字傳遞間的延遲
        yield ch


def send_msg(msg: str, chat: list):
    resp = get_resp()
    chat.append([msg, None])
    return None, chat, resp, Event()


def show_resp(chat: list, resp, event: Event):
    chat[-1][1] = ""
    for ch in resp:
        if event.is_set():
            event.clear()
            chat[-1][1] += " ..."
            chat.append(["冷靜！", "好吧"])
            yield chat
            break

        chat[-1][1] += ch
        yield chat


def stop_show(event: Event):
    event.set()


with gr.Blocks() as demo:
    event = gr.State(None)
    resp = gr.State(None)
    chat = gr.Chatbot([[None, "喵！"]], label="喵星人", height=230)
    msg = gr.Textbox(label="學貓叫")
    stop = gr.Button("冷靜！")

    send_inn, send_out = [msg, chat], [msg, chat, resp, event]
    show_inn, show_out = [chat, resp, event], [chat]
    msg.submit(send_msg, send_inn, send_out).then(show_resp, show_inn, show_out)
    stop.click(stop_show, event, queue=False)

demo.queue().launch()
