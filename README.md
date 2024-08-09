# LLM Note 補充資源

## 介紹

本專案整理了《LLM 大型語言模型的絕世祕笈：27 路獨步劍法，帶你闖蕩生成式 AI 的五湖四海》書中提及的程式碼資源，此書內容改編自第 15 屆 iThome 鐵人賽 AI & Data 組冠軍系列文章[《LLM 學習筆記》](https://ithelp.ithome.com.tw/users/20121763/ironman/6145)，現在可於各大書商通路購買，請各位多多支持！

[博碩](https://www.drmaster.com.tw/Bookinfo.asp?BookID=MP22435) | [天瓏](https://www.tenlong.com.tw/products/9786263339293) | [博客來](https://www.books.com.tw/products/0010996141) | [金石堂](https://www.kingstone.com.tw/basic/2013120710320) | [momo](https://www.momoshop.com.tw/goods/GoodsDetail.jsp?i_code=13087944) | [三民](https://www.sanmin.com.tw/product/index/013473822) | [誠品](https://www.eslite.com/product/10012011762682616069008) | [iThome 系列原文](https://ithelp.ithome.com.tw/users/20121763/ironman/6145)

![Cover](https://www.drmaster.com.tw/Cover/MP22435.png)

## 相關專案

- Ch 1 - [Bigram 語言模型完整程式碼 (Colab)](https://tinyurl.com/llm-note-01)
- Ch 4 - [文字介面聊天範例完整程式碼](OpenAI-API/CLI-Chat-Demo.py)
- Ch 5 - [貓貓塔羅完整程式碼](https://tinyurl.com/llm-note-03)
- Ch 5 - [停止串流範例完整程式碼](Gradio/StopChat.py)
- Ch 8 - [Latex 論文閱讀完整程式碼](https://tinyurl.com/llm-note-06)
- Ch 11 - [HF Transformers 範例程式碼 (Colab)](https://tinyurl.com/llm-note-07)
- Ch 11 - [速度與記憶體評測完整程式碼](HF/HF-Bench.py)
- Ch 13 - [StarCoder 2 使用範例程式碼 (Colab)](https://tinyurl.com/llm-note-08)
- Ch 14 - [簡易量化範例程式碼 (Colab)](https://tinyurl.com/llm-note-09)
- Ch 15 - [速度評測完整程式碼](Utils/BenchSpeed.py)
- Ch 19 - [N-Gram Search 完整程式碼](NGram/NGram-Search.py)
- Ch 21 - [中二技能翻譯完整程式碼](https://tinyurl.com/llm-note-13)

## 內文勘誤

因為書本內容受限於筆者撰文當下的時空背景，因此有些事物變遷無法被紀錄進去。此節收錄書中的內文勘誤，如果發現其他問題也歡迎回報！

### 12.3.1 Breeze & BreeXe
[Breeze](https://huggingface.co/MediaTek-Research) 是由聯發科技集團的 AI 研究單位聯發創新基地（MediaTek Research）所開發的繁體中文語言模型，架構與權重承襲自 Mistral，同樣為 7B 參數量的模型，但是分詞器有針對繁體中文額外擴充詞表。可能是因為參數量並不大的關係，所以能力上也只是普通而已。

但是，開發團隊後來又推出了 [BreeXe-8x7B](https://tinyurl.com/llm-breexe) 的模型，與 Mixtral 一樣採用 MoE 架構，雖然有將近 50B 的參數量，但是能維持 13B 的生成速度，而且生成效果大幅提昇！雖然效果可能不如更大規模的模型，但在同等生成速度下，依然是筆者用過最頂尖的繁體中文模型。模型權重完整開源在 [HF Hub](https://huggingface.co/MediaTek-Research/Breexe-8x7B-Instruct-v0_1) 上，筆者亦有轉換並上傳 [GGUF](https://huggingface.co/PenutChen/Breexe-8x7B-Instruct-v0_1-GGUF) 的版本，此外還有官方 [Demo](https://tinyurl.com/llm-breexe-demo) 網頁可以做測試，推薦各位一定要去用看看！

> 註：修正 BreeXe 未開源的描述。

## 授權

MIT License
