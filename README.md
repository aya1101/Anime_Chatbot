# 📽️ OtakuBot - Movie Recommendation & Chatbot System

OtakuBot là một hệ thống chatbot và gợi ý phim (anime) sử dụng AI, hỗ trợ tiếng Việt, cho phép người dùng:
- Tìm kiếm thông tin phim
- Gợi ý phim theo thể loại hoặc nội dung
- Xem đánh giá, mô tả phim
- Chat hỏi đáp về phim qua giao diện Gradio

## 🚀 Tính năng chính

- **Chatbot phim:** Trả lời tự động các câu hỏi về phim dựa trên dữ liệu phim có sẵn.
- **Gợi ý phim:** Đề xuất phim dựa trên phim yêu thích hoặc thể loại bạn chọn.

## 🛠️ Công nghệ sử dụng

- Python 3.8+
- [LangChain](https://github.com/langchain-ai/langchain)
- [HuggingFace Transformers](https://huggingface.co/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Gradio](https://gradio.app/)
- [scikit-learn](https://scikit-learn.org/)
- [dotenv](https://pypi.org/project/python-dotenv/)

## 📦 Cài đặt

1. **Clone repo:**
    ```sh
    git clone https://github.com/<your-username>/Anime_Chatbot.git
    cd Anime_Chatbot
    ```

2. **Tạo virtual environment (khuyến nghị):**
    ```sh
    python -m venv .venv
    .venv\Scripts\activate  # Windows
    # hoặc
    source .venv/bin/activate  # Linux/Mac
    ```

3. **Cài đặt các thư viện cần thiết:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Tạo file `.env` và điền token HuggingFace:**
    ```env
    HUGGINGFACE_API_TOKEN=your_huggingface_token
    DB_HOST=localhost
    DB_USER=root
    DB_PASSWORD=
    DB_NAME=movies_db
    ```

5. **Chuẩn bị dữ liệu phim:**  
   Đặt file `data_movies.json` vào thư mục `data/` theo cấu trúc mẫu.

## 🏃‍♂️ Chạy chatbot

```sh
python Chatbot/chatbot_local.py
```
Sau đó mở đường link Gradio để chat với bot.

## 📝 Cách sử dụng

- **Chatbot:** Nhập câu hỏi về phim, thể loại, đánh giá, v.v.
- **Gợi ý phim:** Chạy script recommendation để nhận đề xuất phim theo sở thích.

## 📁 Cấu trúc thư mục

```
Recommendation_System/
│
├── Chatbot/
│   ├── chatbot_local.py
│   ├── recomendation_local.py
│   └── ...
├── data/
│   ├── data_movies.json
│   └── stopwords-vi.txt
├── .env.example
├── requirements.txt
└── README.md
```

## ⚠️ Lưu ý bảo mật

- **KHÔNG** commit file `.env` chứa token hoặc mật khẩu lên GitHub.
- Đã có sẵn `.gitignore` để loại trừ `.env`.

## 📚 Tham khảo

- [LangChain Docs](https://python.langchain.com/)
- [HuggingFace Hub](https://huggingface.co/)
- [Gradio Docs](https://gradio.app/docs/)

---

**Chúc bạn trải nghiệm vui vẻ với OtakuBot!**