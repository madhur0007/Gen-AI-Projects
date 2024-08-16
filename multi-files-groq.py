#####single-file-ollama.py is for individual document,while multiple-files-ollama.py functionality for handling multiple documents simultaneously######
###mainmulti-files-groq.py unctionality for handling multiple documents simultaneously ############

import streamlit as st
import pypdfium2 as pdfium
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
import validators
import docx2txt  # For extracting text from Word files
import os
from groq import Groq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()




groq_api_key = os.getenv("GROQ_API_KEY", "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")





class RAGSystem:
    
    def __init__(self, model_name='all-MiniLM-L6-v2', llm_model='mixtral-8x7b-32768'):
        self.model = SentenceTransformer(model_name)
        self.sentences = []
        self.index = None
        self.memory = ConversationBufferWindowMemory(k=5)  # Adjustable memory length
        self.groq_chat = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name=llm_model
        )
        self.conversation = ConversationChain(
            llm=self.groq_chat,
            memory=self.memory
        )

    def extract_text_from_pdfs(self, pdf_files):
        texts = []
        try:
            for pdf_file in pdf_files:
                text = ""
                pdf = pdfium.PdfDocument(pdf_file.read())
                for page_num in range(len(pdf)):
                    page = pdf[page_num]
                    textpage = page.get_textpage()
                    text += textpage.get_text_range()
                texts.append(text)
        except Exception as e:
            st.error(f"Error extracting text from PDFs: {e}")
        return texts

    def extract_text_from_word(self, docx_files):
        texts = []
        try:
            for docx_file in docx_files:
                text = docx2txt.process(docx_file)
                texts.append(text)
        except Exception as e:
            st.error(f"Error extracting text from Word files: {e}")
        return texts

    def extract_text_from_txt(self, txt_files):
        texts = []
        try:
            for txt_file in txt_files:
                text = txt_file.read().decode("utf-8")
                texts.append(text)
        except Exception as e:
            st.error(f"Error extracting text from TXT files: {e}")
        return texts

    def fetch_url_content(self, urls):
        contents = []
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                text = ' '.join([p.get_text() for p in soup.find_all('p')])
                if len(text.split()) < 50:
                    text = ' '.join([div.get_text() for div in soup.find_all('div') if len(div.get_text().split()) > 50])
                contents.append(text)
            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching content from {url}: {e}")
        return contents

    def vectorize_content(self, texts):
        self.sentences = []
        for text in texts:
            sentences = text.split('. ')
            self.sentences.extend(sentences)
        embeddings = self.model.encode(self.sentences)
        self.index = create_faiss_index(embeddings)

    def retrieve_relevant_content(self, query, k=3):
        if self.index is None or not self.sentences:
            raise ValueError("Content has not been embedded yet. Please fetch and embed content first.")
        
        query_embedding = self.model.encode([query])
        D, I = self.index.search(query_embedding, k=k)
        relevant_sentences = [self.sentences[i] for i in I[0]]
        return relevant_sentences

    def generate_answer(self, context, query):
        user_message = f'{context}\n\n{query}'
        response = self.conversation(user_message)
        return response['response']

    def chat_with_llm(self, user_message):
        response = self.conversation(user_message)
        return response['response']


@st.cache_data
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def main():
    st.title("RAG System with PDF, Word, TXT, and URL Support")

    tab1, tab2 = st.tabs(["RAG System", "Direct LLM Chat"])

    with tab1:
        st.subheader("RAG System")
        option = st.radio("Choose input type:", ("PDFs", "Word Files", "TXT Files", "URLs"))
        
        if option == "PDFs":
            pdf_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
            query = st.text_input("Enter your query:")
            if st.button("Submit", key="pdf_rag_submit"):
                if pdf_files and query:
                    rag_system = RAGSystem()
                    texts = rag_system.extract_text_from_pdfs(pdf_files)
                    if not texts:
                        st.error("Failed to extract text from the PDFs.")
                    else:
                        rag_system.vectorize_content(texts)
                        relevant_chunks = rag_system.retrieve_relevant_content(query, k=3)
                        combined_context = ' '.join(relevant_chunks)
                        answer = rag_system.generate_answer(combined_context, query)
                        st.write("Relevant Chunks:", relevant_chunks)
                        st.write("Answer:", answer)
                else:
                    st.warning("Please upload PDF files and enter a query.")
        
        elif option == "Word Files":
            docx_files = st.file_uploader("Upload Word files", type="docx", accept_multiple_files=True)
            query = st.text_input("Enter your query:")
            if st.button("Submit", key="word_rag_submit"):
                if docx_files and query:
                    rag_system = RAGSystem()
                    texts = rag_system.extract_text_from_word(docx_files)
                    if not texts:
                        st.error("Failed to extract text from the Word files.")
                    else:
                        rag_system.vectorize_content(texts)
                        relevant_chunks = rag_system.retrieve_relevant_content(query, k=3)
                        combined_context = ' '.join(relevant_chunks)
                        answer = rag_system.generate_answer(combined_context, query)
                        st.write("Relevant Chunks:", relevant_chunks)
                        st.write("Answer:", answer)
                else:
                    st.warning("Please upload Word files and enter a query.")
        
        elif option == "TXT Files":
            txt_files = st.file_uploader("Upload TXT files", type="txt", accept_multiple_files=True)
            query = st.text_input("Enter your query:")
            if st.button("Submit", key="txt_rag_submit"):
                if txt_files and query:
                    rag_system = RAGSystem()
                    texts = rag_system.extract_text_from_txt(txt_files)
                    if not texts:
                        st.error("Failed to extract text from the TXT files.")
                    else:
                        rag_system.vectorize_content(texts)
                        relevant_chunks = rag_system.retrieve_relevant_content(query, k=3)
                        combined_context = ' '.join(relevant_chunks)
                        answer = rag_system.generate_answer(combined_context, query)
                        st.write("Relevant Chunks:", relevant_chunks)
                        st.write("Answer:", answer)
                else:
                    st.warning("Please upload TXT files and enter a query.")

        elif option == "URLs":
            urls = st.text_area("Enter URLs (one per line):")
            query = st.text_input("Enter your query:")
            if st.button("Submit", key="url_rag_submit"):
                url_list = [url.strip() for url in urls.splitlines() if url.strip()]
                if not all(validators.url(url) for url in url_list):
                    st.error("Please enter valid URLs.")
                    return
                if url_list and query:
                    rag_system = RAGSystem()
                    contents = rag_system.fetch_url_content(url_list)
                    if not contents:
                        st.error("Failed to extract content from the provided URLs.")
                    else:
                        rag_system.vectorize_content(contents)
                        relevant_chunks = rag_system.retrieve_relevant_content(query, k=3)
                        combined_context = ' '.join(relevant_chunks)
                        answer = rag_system.generate_answer(combined_context, query)
                        st.write("Relevant Chunks:", relevant_chunks)
                        st.write("Answer:", answer)
                else:
                    st.warning("Please enter both URLs and a query.")

    with tab2:
        st.subheader("Chat with LLM")
        user_message = st.text_area("Enter your message:")
        if st.button("Send", key="llm_chat"):
            if user_message:
                rag_system = RAGSystem()
                response = rag_system.chat_with_llm(user_message)
                st.write("Response from LLM:", response)
            else:
                st.warning("Please enter a message to chat with the LLM.")

if __name__ == "__main__":
    main()
