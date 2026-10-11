import sqlite3
import uuid
from datetime import datetime

import os
import re
import json
import shutil
import unicodedata
import traceback
from collections import Counter

import gradio as gr
import numpy as np
import torch
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# STUDYMATE AI CONFIGURATION
# ============================================================

APP_TITLE = "StudyMate AI"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ANSWER_MODEL_NAME = "google/flan-t5-small"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
INDEX_PATH = os.path.join(STORAGE_DIR, "studymate_index.faiss")
CHUNKS_PATH = os.path.join(STORAGE_DIR, "chunks.json")
META_PATH = os.path.join(STORAGE_DIR, "metadata.json")

DEVICE = torch.device("cpu")
os.makedirs(STORAGE_DIR, exist_ok=True)

print("Loading StudyMate AI models...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

answer_tokenizer = AutoTokenizer.from_pretrained(ANSWER_MODEL_NAME)
answer_model = AutoModelForSeq2SeqLM.from_pretrained(ANSWER_MODEL_NAME)
answer_model.to(DEVICE)
answer_model.eval()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

knowledge = None

print("StudyMate AI models loaded.")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.encode("utf-8", errors="replace").decode("utf-8")

    text = "".join(
        char if char in "\n\t" or ord(char) >= 32 else " "
        for char in text
    )

    return re.sub(r"[ \t]+", " ", text).strip()


def tokenize_words(text):
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


# ============================================================
# PERSISTENT CHAT HISTORY
# ============================================================

DB_PATH = os.path.join(STORAGE_DIR, "studymate.db")


def init_database():
    os.makedirs(STORAGE_DIR, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
                    ON DELETE CASCADE
            )
        """)

        conn.execute("PRAGMA foreign_keys = ON")


def create_conversation(title="New chat"):
    chat_id = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO conversations
               (id, title, created_at, updated_at)
               VALUES (?, ?, ?, ?)""",
            (chat_id, title, now, now)
        )

    return chat_id


def save_message(chat_id, role, content, sources=""):
    now = datetime.now().isoformat(timespec="seconds")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO messages
               (conversation_id, role, content, sources, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (chat_id, role, content, sources, now)
        )

        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, chat_id)
        )


def get_conversations():
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute(
            """SELECT id, title FROM conversations
               ORDER BY updated_at DESC"""
        ).fetchall()


def get_messages(chat_id):
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute(
            """SELECT role, content, sources
               FROM messages
               WHERE conversation_id = ?
               ORDER BY id ASC""",
            (chat_id,)
        ).fetchall()


def rename_conversation(chat_id, title):
    title = title.strip()[:80]

    if not title:
        return

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """UPDATE conversations
               SET title = ?, updated_at = ?
               WHERE id = ?""",
            (title, datetime.now().isoformat(timespec="seconds"), chat_id)
        )


def delete_conversation(chat_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            "DELETE FROM messages WHERE conversation_id = ?",
            (chat_id,)
        )
        conn.execute(
            "DELETE FROM conversations WHERE id = ?",
            (chat_id,)
        )


# ============================================================
# CHAT SIDEBAR HELPERS
# ============================================================

def get_chat_choices():
    conversations = get_conversations()

    return [
        (title, chat_id)
        for chat_id, title in conversations
    ]


def create_new_chat():
    chat_id = create_conversation("New chat")
    choices = get_chat_choices()

    return (
        chat_id,
        gr.update(choices=choices, value=chat_id),
        [],
        "",
        "New conversation created."
    )


def open_saved_chat(chat_id):
    if not chat_id:
        return [], None

    rows = get_messages(chat_id)
    history = []

    for role, content, sources in rows:
        if role == "user":
            history.append({
                "role": "user",
                "content": content
            })
        elif role == "assistant":
            combined = content

            if sources:
                combined += "\n\n" + sources

            history.append({
                "role": "assistant",
                "content": combined
            })

    return history, chat_id



def send_chat_message(message, history, chat_id):
    message = (message or "").strip()
    history = list(history or [])

    if not message:
        return (
            history, "", chat_id,
            gr.update(choices=get_chat_choices(), value=chat_id),
            "Enter a question first."
        )

    if knowledge is None:
        history.extend([
            {"role": "user", "content": message},
            {
                "role": "assistant",
                "content": "Please upload and process your PDFs first."
            }
        ])

        return (
            history, "", chat_id,
            gr.update(choices=get_chat_choices(), value=chat_id),
            "Process study material before asking questions."
        )

    try:
        if not chat_id:
            title = message[:45].strip()
            if len(message) > 45:
                title += "..."

            chat_id = create_conversation(title)

        # Set the title from the first question.
        if not get_messages(chat_id):
            title = message[:45].strip()
            if len(message) > 45:
                title += "..."
            rename_conversation(chat_id, title)

        # Save the user's message.
        save_message(chat_id, "user", message)

        # Generate the answer using your existing RAG pipeline.
        answer, sources = answer_question(message)

        # Save the answer and its source references.
        save_message(chat_id, "assistant", answer, sources)

        history.extend([
            {"role": "user", "content": message},
            {
                "role": "assistant",
                "content": answer + (
                    "\n\n" + sources if sources else ""
                )
            }
        ])

        return (
            history,
            "",
            chat_id,
            gr.update(
                choices=get_chat_choices(),
                value=chat_id
            ),
            "Conversation saved."
        )

    except Exception as exc:
        traceback.print_exc()

        history.append({
            "role": "assistant",
            "content": f"Sorry, an error occurred: {exc}"
        })

        return (
            history,
            "",
            chat_id,
            gr.update(
                choices=get_chat_choices(),
                value=chat_id
            ),
            f"Message processing failed: {exc}"
        )


def rename_selected_chat(chat_id, new_title):
    if not chat_id:
        return (
            gr.update(choices=get_chat_choices()),
            "Select a conversation to rename."
        )

    if not (new_title or "").strip():
        return (
            gr.update(choices=get_chat_choices()),
            "Enter a title first."
        )

    rename_conversation(chat_id, new_title.strip()[:80])

    return (
        gr.update(
            choices=get_chat_choices(),
            value=chat_id
        ),
        "Conversation renamed."
    )


def delete_selected_chat(chat_id):
    if not chat_id:
        return (
            [],
            None,
            gr.update(choices=get_chat_choices(), value=None),
            "Select a conversation to delete."
        )

    delete_conversation(chat_id)

    return (
        [],
        None,
        gr.update(choices=get_chat_choices(), value=None),
        "Conversation deleted."
    )

init_database()


# ============================================================
# PERSISTENCE: SAVE AND LOAD KNOWLEDGE BASE
# ============================================================

def save_knowledge_base():
    """Save the FAISS index and chunk metadata locally."""

    if knowledge is None:
        return

    faiss.write_index(knowledge["index"], INDEX_PATH)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            knowledge["chunks"],
            file,
            ensure_ascii=False,
            indent=2
        )

    metadata = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "answer_model": ANSWER_MODEL_NAME,
        "sources": knowledge["sources"],
        "chunk_count": len(knowledge["chunks"]),
        "embedding_dimension": knowledge["index"].d
    }

    with open(META_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("Knowledge base saved to storage/.")


def load_knowledge_base():
    """Restore a previously saved knowledge base."""

    global knowledge

    if not all(
        os.path.exists(path)
        for path in [INDEX_PATH, CHUNKS_PATH, META_PATH]
    ):
        return False

    try:
        with open(META_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        if metadata.get("embedding_model") != EMBEDDING_MODEL_NAME:
            print("Saved index uses a different embedding model.")
            return False

        index = faiss.read_index(INDEX_PATH)

        if index.d != embedding_model.get_sentence_embedding_dimension():
            print("Saved index has an incompatible embedding dimension.")
            return False

        with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
            chunks = json.load(file)

        if index.ntotal != len(chunks):
            print("Saved index and chunk count do not match.")
            return False

        if not chunks:
            return False

        knowledge = {
            "index": index,
            "chunks": chunks,
            "sources": metadata.get("sources", [])
        }

        print(f"Restored {len(chunks)} chunks from saved knowledge base.")
        return True

    except Exception:
        print("Could not restore saved knowledge base.")
        traceback.print_exc()
        knowledge = None
        return False


# ============================================================
# PDF PROCESSING
# ============================================================

def process_pdfs(files):
    global knowledge

    if not files:
        return "Please upload at least one PDF file."

    documents = []
    uploaded_names = []
    errors = []

    for file_path in files:
        try:
            if not str(file_path).lower().endswith(".pdf"):
                errors.append(f"Skipped non-PDF file: {file_path}")
                continue

            filename = os.path.basename(str(file_path))
            reader = PdfReader(str(file_path))

            uploaded_names.append(filename)

            for page_number, page in enumerate(reader.pages, start=1):
                text = clean_text(page.extract_text() or "")

                if text:
                    documents.append({
                        "source": filename,
                        "page": page_number,
                        "text": text
                    })

        except Exception as exc:
            errors.append(f"{file_path}: {exc}")

    if not documents:
        message = (
            "No readable text was extracted. "
            "If the PDF contains scanned images, OCR may be required."
        )

        if errors:
            message += "\n\n" + "\n".join(errors)

        return message

    chunks = []

    for document in documents:
        for part in splitter.split_text(document["text"]):
            if part.strip():
                chunks.append({
                    "source": document["source"],
                    "page": document["page"],
                    "text": part
                })

    if not chunks:
        return "No text chunks could be created."

    try:
        status_update = (
            f"Extracted text from {len(documents)} pages.\n"
            f"Creating embeddings for {len(chunks)} chunks..."
        )

        print(status_update)

        texts = [item["text"] for item in chunks]

        vectors = embedding_model.encode(
            texts,
            batch_size=8,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        vectors = np.asarray(vectors, dtype=np.float32)

        new_index = faiss.IndexFlatIP(vectors.shape[1])
        new_index.add(vectors)

        # Replace the active knowledge base only after successful indexing.
        knowledge = {
            "index": new_index,
            "chunks": chunks,
            "sources": sorted(set(uploaded_names))
        }

        save_knowledge_base()

        message = (
            "## PDF processing completed successfully!\n\n"
            f"- **PDF files:** {len(set(uploaded_names))}\n"
            f"- **Pages containing text:** {len(documents)}\n"
            f"- **Text chunks:** {len(chunks)}\n"
            f"- **FAISS vectors:** {new_index.ntotal}\n"
            f"- **Embedding dimensions:** {new_index.d}\n\n"
            "Your knowledge base has been saved locally. "
            "It can be restored the next time you start StudyMate AI."
        )

        if errors:
            message += "\n\n### Warnings\n" + "\n".join(
                f"- {error}" for error in errors
            )

        return message

    except Exception as exc:
        traceback.print_exc()
        return f"Error while processing documents: {exc}"


# ============================================================
# IMPROVED RETRIEVAL
# ============================================================

def retrieve_chunks(question, top_k=4):
    if knowledge is None:
        return []

    query_vector = embedding_model.encode(
        [clean_text(question)],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype(np.float32)

    # Retrieve a wider candidate pool before reranking.
    candidate_count = min(10, knowledge["index"].ntotal)

    scores, ids = knowledge["index"].search(
        query_vector,
        candidate_count
    )

    query_terms = tokenize_words(question)
    candidates = []

    for score, chunk_id in zip(scores[0], ids[0]):
        chunk_id = int(chunk_id)

        if chunk_id < 0 or chunk_id >= len(knowledge["chunks"]):
            continue

        chunk = knowledge["chunks"][chunk_id].copy()
        chunk_terms = tokenize_words(chunk["text"])

        if query_terms:
            lexical_score = (
                len(query_terms.intersection(chunk_terms))
                / len(query_terms)
            )
        else:
            lexical_score = 0.0

        semantic_score = (float(score) + 1.0) / 2.0

        # Combine semantic similarity with keyword overlap.
        combined_score = (
            0.75 * semantic_score
            + 0.25 * lexical_score
        )

        chunk["similarity"] = float(score)
        chunk["relevance"] = combined_score
        candidates.append(chunk)

    candidates.sort(
        key=lambda item: item["relevance"],
        reverse=True
    )

    # Prefer distinct pages when enough candidates are available.
    selected = []
    seen_pages = set()

    for item in candidates:
        page_key = (item["source"], item["page"])

        if page_key not in seen_pages:
            selected.append(item)
            seen_pages.add(page_key)

        if len(selected) >= top_k:
            break

    # Fall back to additional chunks if all candidates came from one page.
    if len(selected) < min(top_k, len(candidates)):
        selected_ids = {
            (item["source"], item["page"], item["text"])
            for item in selected
        }

        for item in candidates:
            key = (item["source"], item["page"], item["text"])

            if key not in selected_ids:
                selected.append(item)
                selected_ids.add(key)

            if len(selected) >= top_k:
                break

    return selected


# ============================================================
# ANSWER GENERATION
# ============================================================

def answer_question(question):
    if not question or not question.strip():
        return (
            "### Enter a question\n\nPlease type a question about your study notes.",
            ""
        )

    if knowledge is None:
        return (
            "### No study material loaded\n\n"
            "Upload your PDFs and process them first.",
            ""
        )

    try:
        retrieved = retrieve_chunks(question, top_k=4)

        if not retrieved:
            return "No relevant passages were found.", ""

        context_parts = []

        for item in retrieved[:3]:
            context_parts.append(
                f"Page {item['page']}:\n{item['text'][:650]}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""You are StudyMate AI, a student study assistant.

Use ONLY the study notes below to answer the question.

Write a useful answer in simple English:
1. Start with a direct definition or answer.
2. Explain the main idea in complete sentences.
3. Include important points or examples only when supported by the notes.
4. Use a short numbered list when appropriate.
5. Never invent facts or claim that a topic is exam-important without evidence.
6. If the notes do not answer the question, state that clearly.

STUDY NOTES:
{context}

QUESTION:
{question}

ANSWER:
"""

        inputs = answer_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            output = answer_model.generate(
                **inputs,
                max_new_tokens=150,
                num_beams=4,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )

        answer = answer_tokenizer.decode(
            output[0],
            skip_special_tokens=True
        ).strip()

        if not answer or len(answer.split()) < 5:
            answer = (
                "The model generated an incomplete answer. "
                "Here is the most relevant passage from your notes:\n\n"
                + retrieved[0]["text"]
            )

        # Format the website answer.
        answer_markdown = (
            "### StudyMate AI Answer\n\n"
            + answer
        )

        # Show distinct source pages in the browser.
        source_lines = []
        seen_sources = set()

        for item in retrieved:
            key = (item["source"], item["page"])

            if key not in seen_sources:
                seen_sources.add(key)
                source_lines.append(
                    f"- **{item['source']}** — page {item['page']}"
                )

        sources_markdown = (
            "### Retrieved Sources\n\n"
            + "\n".join(source_lines)
        )

        # These values are rendered in Gradio, not printed to terminal.
        return answer_markdown, sources_markdown

    except Exception as exc:
        traceback.print_exc()

        return (
            f"### Answer generation error\n\n{str(exc)}",
            ""
        )


# ============================================================
# CLEAR KNOWLEDGE BASE
# ============================================================

def clear_knowledge():
    global knowledge
    knowledge = None

    for path in [INDEX_PATH, CHUNKS_PATH, META_PATH]:
        if os.path.exists(path):
            os.remove(path)

    return (
        "Knowledge base and saved index cleared. Upload and process PDFs again.",
        "",
        ""
    )



# ============================================================
# CHATGPT-STYLE STUDYMATE AI FRONTEND
# ============================================================

CUSTOM_CSS = """
.gradio-container {
    max-width: 1450px !important;
    margin: auto !important;
}

#sidebar {
    min-width: 250px;
}

#chat-title {
    text-align: center;
}

#chat-window {
    min-height: 420px;
}

footer {
    visibility: visible;
}
"""

with gr.Blocks(
    title="StudyMate AI",
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS
) as demo:

    current_chat_id = gr.State(None)

    gr.Markdown(
        "# StudyMate AI\n"
        "Your AI-powered student study assistant",
        elem_id="chat-title"
    )

    with gr.Row(equal_height=False):

        # ---------------- SIDEBAR ----------------

        with gr.Column(scale=1, min_width=250, elem_id="sidebar"):

            gr.Markdown("### Your conversations")

            new_chat_button = gr.Button(
                "+ New Chat",
                variant="primary"
            )

            chat_selector = gr.Dropdown(
                label="Recent Chats",
                choices=get_chat_choices(),
                value=None,
                interactive=True,
                filterable=True
            )

            gr.Markdown("### Manage Chat")

            rename_input = gr.Textbox(
                label="New conversation title",
                placeholder="Enter a new title",
                lines=1
            )

            rename_button = gr.Button("Rename Chat")
            delete_button = gr.Button(
                "Delete Selected Chat",
                variant="stop"
            )

            chat_status = gr.Markdown(
                "Your saved conversations will appear here."
            )

            gr.Markdown("---")

            with gr.Accordion("Upload Study Material", open=False):

                pdf_input = gr.File(
                    label="Upload subject notes (PDF)",
                    file_types=[".pdf"],
                    file_count="multiple",
                    type="filepath"
                )

                process_button = gr.Button(
                    "Process Uploaded PDFs",
                    variant="primary"
                )

                process_status = gr.Markdown(
                    "Upload and process PDFs to begin."
                )

                process_button.click(
                    fn=process_pdfs,
                    inputs=[pdf_input],
                    outputs=[process_status]
                )

        # ---------------- MAIN CHAT ----------------

        with gr.Column(scale=4, min_width=400):

            gr.Markdown("### Ask StudyMate AI")

            chatbot = gr.Chatbot(
                label="Conversation",
                #type="messages",
                height=500,
                #buttons=["copy"],
                placeholder=(
                    "Welcome to StudyMate AI!\n\n"
                    "Start a new chat and ask a question about your notes."
                ),
                elem_id="chat-window"
            )

            with gr.Row():
                message_input = gr.Textbox(
                    label="",
                    placeholder="Ask a question about your study notes...",
                    lines=2,
                    scale=5
                )

                send_button = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1
                )

            gr.Markdown(
                "StudyMate AI answers using your processed study notes. "
                "Check the cited pages against your original PDFs."
            )

    # ---------------- CREATE NEW CHAT ----------------

    new_chat_button.click(
        fn=create_new_chat,
        inputs=[],
        outputs=[
            current_chat_id,
            chat_selector,
            chatbot,
            message_input,
            chat_status
        ]
    )

    # ---------------- OPEN SAVED CHAT ----------------

    chat_selector.change(
        fn=open_saved_chat,
        inputs=[chat_selector],
        outputs=[chatbot, current_chat_id]
    )

    # ---------------- SEND MESSAGE ----------------


    send_outputs = [
        chatbot,
        message_input,
        current_chat_id,
        chat_selector,
        chat_status
    ]

    send_button.click(
        fn=send_chat_message,
        inputs=[
            message_input,
            chatbot,
            current_chat_id
        ],
        outputs=send_outputs
    )

    message_input.submit(
        fn=send_chat_message,
        inputs=[
            message_input,
            chatbot,
            current_chat_id
        ],
        outputs=send_outputs
    )


    # ---------------- RENAME CHAT ----------------

    rename_button.click(
        fn=rename_selected_chat,
        inputs=[current_chat_id, rename_input],
        outputs=[chat_selector, chat_status]
    )

    # ---------------- DELETE CHAT ----------------

    delete_button.click(
        fn=delete_selected_chat,
        inputs=[current_chat_id],
        outputs=[
            chatbot,
            current_chat_id,
            chat_selector,
            chat_status
        ]
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    # Restore the existing FAISS index if your persistence
    # functions are present in app.py.
    if "load_knowledge_base" in globals():
        load_knowledge_base()

    print("Launching StudyMate AI...")
    demo.launch(inbrowser=True)