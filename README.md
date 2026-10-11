# 📚 StudyMate AI Agent

**An AI-powered study assistant for learning from PDF documents using Generative AI.**

StudyMate AI Agent is a Python-based web application that helps students understand their study materials through natural-language question answering. Users can upload PDF documents, ask questions about their content, and receive AI-generated answers supported by relevant document sources.

## ✨ Features

- 📄 **PDF Upload:** Upload study notes, textbooks, and educational documents.
- 🤖 **AI Question Answering:** Ask questions about uploaded study materials.
- 🔍 **Semantic Search:** Find relevant passages using sentence embeddings.
- 📌 **Source References:** Display relevant document sources alongside answers.
- 💬 **Interactive Chat:** Communicate with the AI study assistant.
- 🗂️ **Chat History:** Access previous conversations.
- ✏️ **Chat Management:** Create, rename, and delete conversations.
- 💾 **Persistent Chat Storage:** Store conversation history using SQLite.
- 🌐 **Web Interface:** Access the application through a browser using Gradio.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| Gradio | Web interface |
| PyTorch | Deep learning |
| Hugging Face Transformers | Language model integration |
| FLAN-T5-small | Text generation |
| Sentence Transformers | Text embeddings |
| FAISS CPU | Vector similarity search |
| PyPDF | PDF text extraction |
| LangChain Text Splitters | Document chunking |
| SQLite | Chat history storage |

## ⚙️ System Workflow

1. **PDF Upload:** The user uploads a study document.
2. **Text Extraction:** Text is extracted from the PDF.
3. **Text Chunking:** The extracted text is divided into smaller chunks.
4. **Embedding Generation:** Sentence Transformers generates embeddings for the chunks.
5. **Vector Indexing:** FAISS stores the embeddings for similarity-based retrieval.
6. **Question Input:** The user submits a question through the Gradio interface.
7. **Context Retrieval:** The application retrieves relevant document chunks.
8. **Answer Generation:** The language model generates a response using the retrieved context.
9. **Source Display:** Relevant source information is displayed with the answer.

## 📂 Project Structure

```text
StudyMate-AI-Agent/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── storage/          # Runtime data, if created by the application
```

*The `storage/` directory may be created at runtime and may be excluded from Git. The exact files and directories depend on the application's implementation.*

## 🚀 Installation and Setup

### Prerequisites

- Python 3.11 or a compatible Python version
- Git
- Visual Studio Code or another code editor

### Step 1: Clone the Repository

```bash
git clone https://github.com/SirsodeRaj/StudyMate-AI-Agent.git
cd StudyMate-AI-Agent
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

For Windows Command Prompt:

```bat
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
python app.py
```

Open the local URL displayed in the terminal to access the StudyMate AI web interface.

## 📖 How to Use

1. Start the application.
2. Create a new conversation.
3. Upload a PDF containing your study material.
4. Allow the application to process and index the document.
5. Enter a question related to the uploaded content.
6. Read the generated response and available source references.
7. Continue the conversation or start a new chat.

**Tip:** Text-based PDFs generally work better than scanned documents without OCR.

## 🌐 Deployment

StudyMate AI can be run locally and may be deployed to a compatible cloud hosting platform.

**Planned hosting option:** [Hugging Face Spaces](https://huggingface.co/spaces)

The application requires sufficient memory for its machine-learning dependencies and models. Hosting availability and performance depend on the selected hardware, model sizes, and platform limits.

## 🔒 Data and Privacy

- Chat history is stored using SQLite.
- Uploaded PDFs and vector indexes may be stored locally, depending on the implementation.
- Local databases, private documents, virtual environments, and secrets should not be committed to a public repository.
- Cloud deployment requires checking the platform's data persistence and privacy policies.

## 🎯 Project Objectives

- Simplify learning from PDF-based educational materials.
- Enable natural-language question answering over study documents.
- Demonstrate semantic search and retrieval-augmented generation (RAG) concepts.
- Integrate language models with document retrieval.
- Provide an organized interface for student conversations.

## 🔮 Future Enhancements

- Support for additional document formats.
- Improved retrieval and answer accuracy.
- OCR support for scanned PDFs.
- User authentication and personalized study spaces.
- Persistent cloud storage for uploaded documents.
- Export conversations and answers.
- Performance optimization and support for larger language models.

## 👨‍💻 Author

**Raj Sirsode**

GitHub: [@SirsodeRaj](https://github.com/SirsodeRaj)

## 📄 License

No license has been specified for this repository. Add an appropriate license if you intend to permit others to reuse, modify, or distribute the project.

---

**StudyMate AI Agent — Your AI-powered companion for smarter learning.**
