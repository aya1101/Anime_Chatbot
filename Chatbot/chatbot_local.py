import os
import json
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
import gradio as gr
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

class MovieChatbot:
    def __init__(self, json_path: str = "data/data_movies.json"):
        """Khởi tạo chatbot với dữ liệu phim từ file JSON."""
        self.json_path = json_path
        self.vector_store = None
        self.chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.api_key = os.getenv("HUGGINGFACE_API_TOKEN")
        if not self.api_key:
            raise ValueError("Không tìm thấy HUGGINGFACE_API_KEY trong file .env")
        
    def load_movie_data(self) -> List[Dict]:
        """Tải và xử lý dữ liệu phim từ file JSON."""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Chuyển đổi dữ liệu thành định dạng phù hợp
            movies_list = []
            for movie_id, movie_data in data.items():
                movie_text = f"""
                Tên phim: {movie_data['title']}
                Thể loại: {', '.join(movie_data['genre'])}
                Đánh giá: {movie_data['rating'][0]} ({movie_data['rating'][1]} lượt đánh giá)
                Trạng thái: {movie_data['status']}
                Số tập: {movie_data['episodes']}
                Năm phát hành: {movie_data['release year']}
                Mô tả: {movie_data['description']}
                """
                movies_list.append({
                    'text': movie_text,
                    'metadata': {
                        'movie_id': movie_id,
                        'title': movie_data['title'],
                        'genres': movie_data['genre'],
                        'rating': movie_data['rating'],
                        'status': movie_data['status'],
                        'episodes': movie_data['episodes'],
                        'release_year': movie_data['release year']
                    }
                })
            return movies_list
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu: {str(e)}")
            return []

    def create_vector_store(self, save_path: str = "vector_store"):
        """Tạo và lưu vector store từ dữ liệu phim."""
        try:
            movies = self.load_movie_data()
            if not movies:
                return
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            texts = []
            metadatas = []
            for movie in movies:
                chunks = text_splitter.split_text(movie['text'])
                texts.extend(chunks)
                metadatas.extend([movie['metadata']] * len(chunks))
            
            #embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="google/flan-t5-large"
            )
            
            # Tạo vector store
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=embeddings,
                metadatas=metadatas
            )
            
            # Lưu vector store
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self.vector_store.save_local(save_path)
            print(f"Đã lưu vector store tại: {save_path}")
            
        except Exception as e:
            print(f"Lỗi khi tạo vector store: {str(e)}")

    def load_vector_store(self, load_path: str = "vector_store"):
        """Tải vector store đã lưu."""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            self.vector_store = FAISS.load_local(
                load_path, embeddings, allow_dangerous_deserialization=True
            )
            print(f"Đã tải vector store từ: {load_path}")
        except Exception as e:
            print(f"Lỗi khi tải vector store: {str(e)}")

    def setup_chain(self, api_key: str):
            """Thiết lập chuỗi xử lý với LLM."""
            try:
                llm = HuggingFaceHub(
                    repo_id="google/flan-t5-large",  # Thay vì declare-lab/flan-alpaca-large
                    task="text2text-generation",
                    model_kwargs={"temperature": 0.5, "max_length": 512},
                    huggingfacehub_api_token=api_key
                )
                
                # prompt template
                template = """
                Bạn là một trợ lý thông minh chuyên về phim ảnh. Hãy trả lời câu hỏi dựa trên thông tin phim được cung cấp.
                
                Thông tin phim:
                {context}
                
                Câu hỏi: {question}
                
                Trả lời:
                """
                
                prompt = PromptTemplate(
                    template=template,
                    input_variables=["context", "question"]
                )
                
                # Tạo chain
                self.chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=self.vector_store.as_retriever(
                        search_kwargs={"k": 3}
                    ),
                    memory=self.memory,
                    combine_docs_chain_kwargs={"prompt": prompt}
                )
                return True
                
            except Exception as e:
                print(f"Lỗi khi thiết lập chain: {str(e)}")
                return False

    def chat(self, question: str, history: List[List[str]] = None) -> str:
        """Xử lý câu hỏi và trả về câu trả lời."""
        if not self.chain:
            return "Vui lòng thiết lập API key trước khi chat."
        
        try:
            response = self.chain.invoke({"question": question})
            return response['answer']
        except Exception as e:
            return f"Lỗi khi xử lý câu hỏi: {str(e)}"
        
def create_chatbot_interface():
    """Tạo giao diện Gradio cho chatbot."""
    try:
        chatbot = MovieChatbot()
        
        # Tạo vector store mới hoặc tải từ file đã lưu
        if not os.path.exists("vector_store"):
            print("Đang tạo vector store mới...")
            chatbot.create_vector_store()
        else:
            print("Đang tải vector store đã lưu...")
            chatbot.load_vector_store()
        
        # Thiết lập chain 
        if not chatbot.setup_chain(chatbot.api_key):
            raise ValueError("Không thể thiết lập chain với API key")
        
        # Tạo giao diện Gradio
        with gr.Blocks(title="Chatbot Phim", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 🤖 OtakuBot
            Chatbot này có thể trả lời các câu hỏi về phim dựa trên dữ liệu phim có sẵn.
            
            ### Các tính năng:
            - Tìm kiếm thông tin phim
            - Gợi ý phim theo thể loại
            - Xem đánh giá và mô tả phim
            """)
            
            chatbot_interface = gr.ChatInterface(
                fn=chatbot.chat,
                title="Chat với Bot",
                description="Nhập câu hỏi của bạn về anime...",
                examples=[
                    "Kể cho tôi về phim One Piece",
                    "Có những phim boylove nào đang chiếu?",
                    "Phim nào có rating cao nhất?",
                    "Tìm phim có thể loại Học đường"
                ],
                theme=gr.themes.Soft()
            )
        
        return demo
    except Exception as e:
        print(f"Lỗi khi khởi tạo chatbot: {str(e)}")
        return None

def main():
    demo = create_chatbot_interface()
    if demo:
        demo.launch(share=True)
    else:
        print("Không thể khởi tạo chatbot. ")

if __name__ == "__main__":
    main() 