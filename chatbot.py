import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Replace this with your actual API key ---
OpenAI_API_KEY = "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"

# --- Streamlit UI ---
st.header("📘 NoteBot - Ask Questions About Your Notes")

with st.sidebar:
    st.title("Upload Notes")
    file = st.file_uploader("Upload a PDF file", type="pdf")

# --- Extract Text from PDF ---
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    # --- Split text into chunks ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len
    )
    chunks = splitter.split_text(text)

    # --- Create embeddings ---
    embeddings = OpenAIEmbeddings(api_key=OpenAI_API_KEY)

    # --- Store embeddings in FAISS vector database ---
    vector_store = FAISS.from_texts(chunks, embeddings)

    # --- Create retriever ---
    retriever = vector_store.as_retriever()

    # --- Define function to format documents ---
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # --- Define LLM ---
    llm = ChatOpenAI(
        api_key=OpenAI_API_KEY,
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=300
    )

    # --- Define custom prompt ---
    customized_prompt = ChatPromptTemplate.from_template("""
    You are a helpful tutor. Use the provided context to answer the question.
    If the context does not contain the answer, say "I don't know Arpan."

    Context:
    {context}

    Question: {input}
    """)

    # --- Create the chain using LCEL ---
    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | customized_prompt
        | llm
        | StrOutputParser()
    )

    # --- User input ---
    user_query = st.text_input("💬 Ask a question about your notes:")

    if user_query:
        # --- Run the chain ---
        response = chain.invoke(user_query)

        # --- Display the response ---
        st.subheader("🧠 Answer:")
        st.write(response)
