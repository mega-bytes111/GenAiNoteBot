import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain.chains.question_answering import load_qa_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate

# Replace with your actual API key
OpenAI_API_KEY = "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"

st.header("NoteBot")

with st.sidebar:
    st.title("My Notes")
    file = st.file_uploader("Upload notes PDF and start asking questions", type="pdf")

# Extracting the text from pdf file
if file is not None:
    my_pdf = PdfReader(file)
    text = ""
    for page in my_pdf.pages:
        text += page.extract_text()
        # st.write(text)

    # Break text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50, length_function=len
    )
    chunks = splitter.split_text(text)
    # st.write(chunks)

    # Create embeddings
    embeddings = OpenAIEmbeddings(api_key=OpenAI_API_KEY)

    # Create VectorDB & store embeddings
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Get user query
    user_query = st.text_input("Type your query here")

    # Perform semantic search
    if user_query:
        matching_chunks = vector_store.similarity_search(user_query)

        # Define LLM
        llm = ChatOpenAI(
            api_key=OpenAI_API_KEY,
            max_tokens=300,
            temperature=0,
            model="gpt-3.5-turbo"
        )

        # --- Approach 1: Using load_qa_chain() (commented) ---
        # chain = load_qa_chain(llm, chain_type="stuff")
        # output = chain.run(question=user_query, input_documents=matching_chunks)
        # st.write(output)

        # --- Approach 2: Using create_stuff_documents_chain() ---
        customized_prompt = ChatPromptTemplate.from_template(
            """You are my assistant tutor. Answer the question based on the following context 
            and if you did not get the context simply say "I don't know Jenny":

            {context}

            Question: {input}
            """
        )

        chain = create_stuff_documents_chain(llm, customized_prompt)
        output = chain.invoke({"input": user_query, "input_documents": matching_chunks})

        st.write(output)
