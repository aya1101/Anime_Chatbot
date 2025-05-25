import os
import sys
import io

# Cấu hình encoding cho Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from typing import List, Dict
import mysql.connector
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

# Cấu hình kết nối MySQL
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'movies_db')
}

class MovieChatbot:
    def __init__(self):
        """Khởi tạo chatbot với các thành phần cần thiết."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
    def load_data_from_db(self) -> pd.DataFrame:
        """Tải dữ liệu phim từ MySQL database."""
        try:
            print("Đang kết nối đến database...")
            conn = mysql.connector.connect(**DB_CONFIG)
            print("Kết nối database thành công")
            
            query = """
            SELECT title, genres, description, rating_score, rating_count 
            FROM movies
            """
            print("Đang thực thi truy vấn.")
            df = pd.read_sql(query, conn)
            conn.close()
            print(f"Đã tải {len(df)} phim từ database")
            return df
        except mysql.connector.Error as err:
            print(f"Lỗi MySQL: {err}")
            if err.errno == mysql.connector.errorcode.ER_ACCESS_DENIED_ERROR:
                print("Sai username/password")
            elif err.errno == mysql.connector.errorcode.ER_BAD_DB_ERROR:
                print("Database không tồn tại")
            else:
                print(f"Lỗi khác: {err}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Lỗi không xác định: {str(e)}")
            return pd.DataFrame()

    def prepare_documents(self, df: pd.DataFrame) -> List[str]:
        """Chuẩn bị dữ liệu phim thành các đoạn văn bản."""
        documents = []
        for _, row in df.iterrows():
            doc = f"""
            Tên phim: {row['title']}
            Thể loại: {row['genres']}
            Mô tả: {row['description']}
            Điểm đánh giá: {row['rating_score']}
            Số lượt đánh giá: {row['rating_count']}
            """
            documents.append(doc)
        return documents

    def create_vector_store(self, documents: List[str]):
        """Tạo vector store từ các tài liệu."""
        # Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.create_documents(documents)
        
        # Tạo vector store
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        print("Tạo vector store thành công")

    def setup_qa_chain(self):
        """Thiết lập chuỗi QA với LLM."""
        # Tạo prompt template
        template = """
        Bạn là một trợ lý thông minh chuyên về phim anime. 
        Sử dụng thông tin từ cơ sở dữ liệu phim để trả lời câu hỏi.
        Nếu không có thông tin phù hợp, hãy nói rằng bạn không biết.

        Context: {context}
        Chat History: {chat_history}
        Human: {question}
        Assistant: """

        PROMPT = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template
        )

        # Khởi tạo LLM từ HuggingFace Hub
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={"temperature": 0.5, "max_length": 512},
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
        )

        # Tạo chuỗi QA
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )

    def initialize(self):
        """Khởi tạo toàn bộ hệ thống chatbot."""
        print("Đang khởi tạo chatbot...")
        
        # Tải dữ liệu
        df = self.load_data_from_db()
        if df.empty:
            raise Exception("Không thể tải dữ liệu từ database.")
        
        # Chuẩn bị tài liệu
        documents = self.prepare_documents(df)
        
        # Tạo vector store
        self.create_vector_store(documents)
        
        # Thiết lập QA chain
        self.setup_qa_chain()
        
        print("Chatbot đã sẵn sàng!")

    def chat(self, question: str) -> str:
        """Xử lý câu hỏi và trả về câu trả lời."""
        if not self.qa_chain:
            return "Chatbot chưa được khởi tạo. Vui lòng gọi initialize() trước."
        
        try:
            response = self.qa_chain({"question": question})
            return response["answer"]
        except Exception as e:
            return f"Có lỗi xảy ra: {str(e)}"

def main():
    # Khởi tạo chatbot
    chatbot = MovieChatbot()
    chatbot.initialize()
    
    print("\nChatbot đã sẵn sàng! Gõ 'quit' để thoát.")
    print("Bạn có thể hỏi về các phim anime, ví dụ:")
    print("- 'Kể cho tôi về phim One Piece'")
    print("- 'Những phim hành động nào hay?'")
    print("- 'Phim nào có rating cao nhất?'")
    
    while True:
        user_input = input("\nBạn: ").strip()
        if user_input.lower() == 'quit':
            break
            
        response = chatbot.chat(user_input)
        print(f"\nChatbot: {response}")

if __name__ == "__main__":
    main() 