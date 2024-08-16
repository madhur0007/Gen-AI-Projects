## RAG System with PDF, Word, TXT, and URL Support
This project implements a Retrieval-Augmented Generation (RAG) system using Python. It supports input from multiple sources such as PDF, Word, TXT files, and URLs. It combines retrieval techniques with language models (LLMs) for generating responses to user queries based on relevant content from the documents. The project utilizes Groq API for the LLM integration and can handle multiple files simultaneously.

Features
## RAG System:

Extracts and processes text from multiple document formats (PDF, DOCX, TXT) and from URLs.
Vectorizes the content using sentence embeddings.
Retrieves the most relevant content to answer a user query using FAISS (similarity search).
Generates answers by combining the retrieved content and querying the LLM.
Direct LLM Chat:

Allows users to directly chat with the integrated LLM model, powered by the Groq API.
Prerequisites
Before running this project, ensure you have installed the following:

## Python 3.8+
Required Python libraries (see the requirements.txt file)
Installation
Clone the repository:

bash
## Copy code
git clone https://github.com/your-username/rag-system.git
cd rag-system
Install dependencies:
Make sure to install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Set up Groq API key:
Create a .env file in the root directory and add your Groq API key:

bash
Copy code
GROQ_API_KEY=your_groq_api_key
Usage
Run the Streamlit app:
To start the app, run the following command in the terminal:

bash
Copy code
streamlit run mainmulti-files-groq.py
Using the App:

## RAG System Tab:
Select the type of input (PDF, Word, TXT, or URLs).
Upload your files or paste URLs.
Enter a query to find relevant content and get answers based on the documents.
Direct LLM Chat Tab:
Enter a message to chat directly with the language model.
File Structure
mainmulti-files-groq.py: Main script that handles multiple file types for the RAG system.
requirements.txt: File listing all the required Python libraries for this project.
.env: Configuration file for storing your Groq API key (must be created by the user).
Tabs:
## RAG System: Upload files or URLs, query the system, and receive answers based on relevant document content.
Direct LLM Chat: Chat with the LLM directly without the document retrieval process.
Supported File Formats
PDF: Extracts text from uploaded PDF documents.
Word (DOCX): Extracts text from Microsoft Word files.
TXT: Extracts text from plain text files.
URLs: Extracts text from web pages.
Key Dependencies
Streamlit: For the web app interface.
pypdfium2: To extract text from PDF files.
sentence-transformers: For generating sentence embeddings.
faiss-cpu: For similarity search on embeddings.
Ollama: For language model integration.
Groq API: For the Groq language model interaction.
docx2txt: For extracting text from DOCX files.
requests & BeautifulSoup: For fetching and parsing web content.
Contributing
Feel free to submit a pull request if you want to contribute to this project. Make sure your code is properly documented and tested.

License
This project is licensed under the MIT License. See the LICENSE file for more details.
