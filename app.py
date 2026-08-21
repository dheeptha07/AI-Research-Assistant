import os
import hashlib
import streamlit as st

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# Environment
# -----------------------------

load_dotenv()

VECTOR_DB_PATH = "vectorstore"


# -----------------------------
# Helper Functions
# -----------------------------

def clean_source_text(text):

    replacements = {
        "": "fi",
        "": "ft",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = " ".join(
        text.split()
    )

    return text.strip()


def looks_like_reference(text):

    lower = text.casefold()

    reference_markers = [
        "doi:",
        "http://",
        "https://",
        "in proceedings",
        "elsevier",
        "acm, new york",
        "et al.",
        "bibliography",
        "references"
    ]

    marker_count = sum(
        marker in lower
        for marker in reference_markers
    )

    return marker_count >= 2


def get_metadata_answer(question):

    q = question.strip().casefold()

    # -----------------------------
    # Page Count
    # -----------------------------

    if (
        "how many pages" in q
        or "number of pages" in q
        or "page count" in q
        or "how many page" in q
    ):

        if st.session_state.active_file_name:

            return (
                f"The uploaded PDF contains "
                f"{st.session_state.active_page_count} pages."
            )

        return "No PDF has been uploaded."

    # -----------------------------
    # Chunk Count
    # -----------------------------

    if (
        "how many chunks" in q
        or "number of chunks" in q
        or "chunk count" in q
        or "how many text chunks" in q
    ):

        if st.session_state.active_file_name:

            return (
                f"The document was split into "
                f"{st.session_state.active_chunk_count} text chunks."
            )

        return "No PDF has been uploaded."

    # -----------------------------
    # File Name
    # -----------------------------

    if (
        "what is the file name" in q
        or "what is the filename" in q
        or "what file did i upload" in q
        or "what pdf did i upload" in q
    ):

        if st.session_state.active_file_name:

            return (
                f"The uploaded file is "
                f"{st.session_state.active_file_name}."
            )

        return "No PDF has been uploaded."

    return None


# -----------------------------
# Cached Resources
# -----------------------------

@st.cache_resource
def load_embeddings():

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_llm():

    from langchain_groq import ChatGroq

    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="openai/gpt-oss-20b",
    )


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖"
)


# -----------------------------
# Session State
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_file_name" not in st.session_state:
    st.session_state.active_file_name = None

if "active_file_hash" not in st.session_state:
    st.session_state.active_file_hash = None

if "active_db" not in st.session_state:
    st.session_state.active_db = None

if "active_page_count" not in st.session_state:
    st.session_state.active_page_count = 0

if "active_chunk_count" not in st.session_state:
    st.session_state.active_chunk_count = 0

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.title("🤖 AI Research Assistant")

    st.markdown(
        """
        **About**

        Ask questions about your research PDF
        and get answers using AI-powered
        document retrieval.
        """
    )

    st.divider()

    st.subheader("📄 Document")

    uploaded_file = st.file_uploader(
        "Upload a research PDF",
        type=["pdf"],
        key=f"pdf_uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file is not None:

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

        st.caption(
            "This document is currently active."
        )

    st.subheader("🧠 Embeddings")
    st.write("all-MiniLM-L6-v2")

    st.subheader("⚡ Language Model")
    st.write("OpenAI GPT-OSS 20B")

    st.divider()

    document_info_placeholder = st.container()

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    if st.button(
        "📄 Clear Document",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.session_state.active_file_name = None
        st.session_state.active_file_hash = None
        st.session_state.active_db = None
        st.session_state.active_page_count = 0
        st.session_state.active_chunk_count = 0

        st.session_state.uploader_key += 1

        st.rerun()


# -----------------------------
# Main Title
# -----------------------------

st.title("🤖 AI Research Assistant")


# -----------------------------
# Empty State
# -----------------------------

if st.session_state.active_file_name is None:

    st.info(
        "📄 Upload a research PDF from the sidebar to get started."
    )

    st.markdown(
        """
        **Example questions**

        - What is this paper about?
        - What methodology did the authors use?
        - What were the main findings?
        - What did the authors propose?
        """
    )


# -----------------------------
# Create / Load Vector Database
# -----------------------------

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    file_hash = hashlib.md5(
        file_bytes
    ).hexdigest()

    new_document = (
        st.session_state.active_file_hash != file_hash
        or st.session_state.active_db is None
    )

    if new_document:

        st.session_state.messages = []

        temp_pdf_path = "uploaded_paper.pdf"

        try:

            with st.spinner(
                f"Processing {uploaded_file.name}..."
            ):

                from langchain_community.document_loaders import (
                    PyPDFLoader
                )

                from langchain_community.vectorstores import (
                    FAISS
                )

                embeddings = load_embeddings()

                with open(
                    temp_pdf_path,
                    "wb"
                ) as f:

                    f.write(file_bytes)

                loader = PyPDFLoader(
                    temp_pdf_path
                )

                uploaded_documents = loader.load()

                if not uploaded_documents:

                    st.error(
                        "❌ The PDF could not be read because "
                        "it contains no readable pages."
                    )

                    st.stop()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                uploaded_chunks = (
                    splitter.split_documents(
                        uploaded_documents
                    )
                )

                if not uploaded_chunks:

                    st.error(
                        "❌ No readable text could be extracted "
                        "from this PDF."
                    )

                    st.stop()

                st.session_state.active_page_count = (
                    len(uploaded_documents)
                )

                st.session_state.active_chunk_count = (
                    len(uploaded_chunks)
                )

                db = FAISS.from_documents(
                    uploaded_chunks,
                    embeddings
                )

            st.success(
                f"✅ {uploaded_file.name} is ready — "
                f"{len(uploaded_documents)} pages • "
                f"{len(uploaded_chunks)} chunks"
            )

        except Exception:

            st.error(
                "❌ Unable to process this PDF. "
                "Please upload a valid, readable PDF."
            )

            st.stop()

        st.session_state.active_db = db

        st.session_state.active_file_name = (
            uploaded_file.name
        )

        st.session_state.active_file_hash = (
            file_hash
        )

    else:

        db = st.session_state.active_db

else:

    db = None


# -----------------------------
# Update Sidebar Document Info
# -----------------------------

if st.session_state.active_file_name:

    document_info_placeholder.subheader(
        "📊 Document Information"
    )

    document_info_placeholder.write(
        f"📄 **File:** "
        f"{st.session_state.active_file_name}"
    )

    document_info_placeholder.write(
        f"📑 **Pages:** "
        f"{st.session_state.active_page_count}"
    )

    document_info_placeholder.write(
        f"🧩 **Chunks:** "
        f"{st.session_state.active_chunk_count}"
    )

else:

    document_info_placeholder.subheader(
        "📊 Document Information"
    )

    document_info_placeholder.write(
        "No document uploaded."
    )


# -----------------------------
# Active Retriever
# -----------------------------

retriever = None

if db is not None:

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5
        }
    )


# -----------------------------
# Query Rewriting Prompt
# -----------------------------

query_prompt = ChatPromptTemplate.from_template(
    """
You are the query-planning component of a research-paper assistant.

Your job is NOT to answer the user's question.

Rewrite the latest user question into the best possible search query
for retrieving the exact evidence needed from the research paper.

Use the conversation history to resolve pronouns and references such as:
they, them, their, it, this, that, these, those.

IMPORTANT SEARCH RULES:

1. For broad questions such as:
   - "What is this paper about?"
   - "What is the paper about?"
   - "What is the main idea?"
   - "What is the study about?"
   prioritize the paper's title, abstract, introduction, purpose,
   objectives, findings, and conclusion.

2. For questions about contributions, proposals, recommendations,
   solutions, or what the authors suggested:
   explicitly search for terms such as:
   proposed, propose, recommendations, design recommendations,
   contributions, implications, suggested, advocated, future directions,
   infrastructural reform, technological design, implementation.

3. For questions asking what "they" proposed or suggested,
   identify who "they" refers to from the conversation and include
   the subject of the proposal explicitly in the rewritten query.

4. Preserve important domain-specific terms from the conversation.

5. Do not answer the question.

Return ONLY the rewritten search query.
Do not add explanations or quotation marks.

Conversation history:
{history}

Latest question:
{question}
"""
)


# -----------------------------
# Answer Generation Prompt
# -----------------------------

answer_prompt = ChatPromptTemplate.from_template(
    """
You are an AI Research Assistant.

Answer ONLY using the context from the research paper below.

If the answer is not present in the context, say:

"I couldn't find that information in the PDF."

Do not make up information.

IMPORTANT:

- Use the terminology and meanings established in the retrieved context.
- Do not expand or reinterpret abbreviations using general knowledge.
- If an abbreviation has an ambiguous meaning, preserve the abbreviation
  exactly as supported by the context.
- Do not introduce technical terms that are not supported by the context.
- When summarizing the paper, prioritize the paper's own description
  of its purpose, methods, findings, and contributions.
- For follow-up questions, use the conversation context to understand
  what the user is referring to, but base the actual answer only on
  the retrieved paper content.

Context:
{context}

Question:
{question}
"""
)


# -----------------------------
# Display Chat History
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )

        if (
            "sources" in message
            and message["sources"]
        ):

            st.markdown(
                "**📚 Sources**"
            )

            document_name = message.get(
                "document",
                "paper.pdf"
            )

            for source in message["sources"]:

                with st.expander(
                    f"📄 {document_name} — "
                    f"Page {source['page']}"
                ):

                    st.write(
                        source["text"][:400] + "..."
                        if len(source["text"]) > 400
                        else source["text"]
                    )


# -----------------------------
# User Input
# -----------------------------

question = st.chat_input(
    "Ask a question about your PDF"
)


# -----------------------------
# Process Question
# -----------------------------

if question:

    search_question = (
        question.strip().casefold()
    )

    # -----------------------------
    # Save User Message
    # -----------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # -----------------------------
    # Display User Message
    # -----------------------------

    with st.chat_message(
        "user"
    ):

        st.write(
            question
        )

    # -----------------------------
    # Metadata Questions
    # -----------------------------

    metadata_answer = get_metadata_answer(
        search_question
    )

    if metadata_answer:

        document_name = (
            uploaded_file.name
            if uploaded_file is not None
            else "paper.pdf"
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": metadata_answer,
                "sources": [],
                "document": document_name
            }
        )

        with st.chat_message(
            "assistant"
        ):

            st.write(
                metadata_answer
            )

        st.stop()

    # -----------------------------
    # Require Uploaded PDF
    # -----------------------------

    if retriever is None:

        answer = (
            "Please upload a research PDF before "
            "asking questions."
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": [],
                "document": "paper.pdf"
            }
        )

        with st.chat_message(
            "assistant"
        ):

            st.write(
                answer
            )

        st.stop()

    # -----------------------------
    # Load LLM Only When Needed
    # -----------------------------

    llm = load_llm()

    # -----------------------------
    # Build Conversation History
    # -----------------------------

    previous_messages = (
        st.session_state.messages[:-1]
    )

    history = ""

    for message in previous_messages:

        history += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )

    # -----------------------------
    # Rewrite Search Query
    # -----------------------------

    with st.spinner(
        "Understanding your question..."
    ):

        query_messages = (
            query_prompt.format_messages(
                history=history,
                question=search_question
            )
        )

        rewritten_query_response = (
            llm.invoke(
                query_messages
            )
        )

        search_query = (
            rewritten_query_response.content.strip()
        )

    # -----------------------------
    # Retrieve Relevant Documents
    # -----------------------------

    with st.spinner(
        "Searching the research paper..."
    ):

        docs = retriever.invoke(
            search_query
        )

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

    # -----------------------------
    # Generate Answer
    # -----------------------------

    with st.spinner(
        "Generating answer..."
    ):

        answer_messages = (
            answer_prompt.format_messages(
                context=context,
                question=question
            )
        )

        response = llm.invoke(
            answer_messages
        )

    # -----------------------------
    # Build Source List
    # -----------------------------

    all_sources = []

    for doc in docs:

        page = doc.metadata.get(
            "page"
        )

        if page is None:
            continue

        cleaned_text = clean_source_text(
            doc.page_content
        )

        all_sources.append(
            {
                "page": page + 1,
                "text": cleaned_text
            }
        )

    # -----------------------------
    # Filter Reference Chunks
    # -----------------------------

    filtered_sources = [
        source
        for source in all_sources
        if not looks_like_reference(
            source["text"]
        )
    ]

    if filtered_sources:

        source_pages = filtered_sources

    else:

        source_pages = all_sources

    # -----------------------------
    # Remove Duplicate Pages
    # -----------------------------

    unique_sources = {}

    for source in source_pages:

        page = source["page"]

        if page not in unique_sources:

            unique_sources[page] = (
                source["text"]
            )

    source_pages = [
        {
            "page": page,
            "text": text
        }
        for page, text
        in sorted(
            unique_sources.items()
        )
    ]

    # -----------------------------
    # Active Document Name
    # -----------------------------

    document_name = (
        uploaded_file.name
        if uploaded_file is not None
        else "paper.pdf"
    )

    # -----------------------------
    # Save Assistant Message
    # -----------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.content,
            "sources": source_pages,
            "document": document_name
        }
    )

    # -----------------------------
    # Display Assistant Message
    # -----------------------------

    with st.chat_message(
        "assistant"
    ):

        st.write(
            response.content
        )

        if source_pages:

            st.markdown(
                "**📚 Sources**"
            )

            for source in source_pages:

                with st.expander(
                    f"📄 {document_name} — "
                    f"Page {source['page']}"
                ):

                    st.write(
                        source["text"][:400] + "..."
                        if len(source["text"]) > 400
                        else source["text"]
                    )