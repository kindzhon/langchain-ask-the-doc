import datetime
import glob
import os
import openai
import streamlit as st
from langchain.chains import ConversationalRetrievalChain,RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter,TokenTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
# Set up
import random
import time
from PyPDF2 import PdfReader
os.environ['OPENAI_API_BASE'] = 'https://api.chatanywhere.com.cn/v1'  

#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
#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
#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
#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
#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
st.title("PDF文档对话聊天机器人")
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

template = """
请使用以下的上下文回答最后的问题。如果你不知道答案，那就说你不知道，不要试图编造答案。必须使用中文来回答以下问题，回答的内容尽可能详细。
上下文：{context}
问题: {question}
"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

def set_api_key():
    user_api_key = st.sidebar.text_input(
        label="#### 在此填入API key填写完成后回车 ",
        placeholder="Paste your openAI API key, sk-",
        type="password")
    if user_api_key:
        os.environ["OPENAI_API_KEY"] = user_api_key
        openai.api_key = user_api_key
def load_db(pdf, chain_type, k):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()

    embeddings = OpenAIEmbeddings()
    chunks = text_splitter.split_text(text)
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    retriever = knowledge_base.as_retriever(search_type="similarity", search_kwargs={"k": k})
    print(retriever)
    # create a chatbot chain. Memory is managed externally.
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name=llm_name, temperature=0),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}

    )
    return qa
chat_history = []
def main():
    st.sidebar.title("请选择PDF文件")
    pdf_list = st.sidebar.file_uploader("一次性选择一个PDF文件", type="pdf", accept_multiple_files=False)
    if pdf_list is not None:
        st.sidebar.write("文件载入成功，现在可以进行文档问答")
    print(pdf_list)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        #print(message)
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Accept user input
    if prompt := st.chat_input("What is up?"):
        if pdf_list is not None:
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                qa = load_db(pdf_list, "stuff", 4)
                result = qa({"query": prompt, "chat_history": chat_history})
                chat_history.append((prompt, result["result"]))
                assistant_response =result["result"]
                # Simulate stream of response with milliseconds delay
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                #print(st.session_state.messages)
if __name__ == '__main__':
    set_api_key()
    main()
