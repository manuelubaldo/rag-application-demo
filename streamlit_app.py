import time
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.anyscale import Anyscale
from llama_index.embeddings.anyscale import AnyscaleEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.memory import ChatMemoryBuffer
import streamlit as st

model_id = "meta-llama/Llama-2-70b-chat-hf"
embed_model_id = "thenlper/gte-large"
chunk_size = 128
chunk_overlap = 8
accepted_file_types = ["txt", "doc", "docx", "pdf"]

llm = Anyscale(
    model=model_id
)

embeddings = AnyscaleEmbedding(
    model=embed_model_id,
    embed_batch_size=30
)

Settings.llm = llm
Settings.embed_model = embeddings

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        embeddings
    ]
)


def generate_response(query):
    chat_engine = st.session_state["chat_engine"]
    response_stream = chat_engine.stream_chat(query)

    with st.chat_message("assistant"):
        response = st.write_stream(response_stream.response_gen)

    return response


if "messages" not in st.session_state:
    st.session_state.messages = []


file = st.file_uploader(
    "Upload a file", type=accepted_file_types, key="file")
if file is not None:
    # save file to ./data/ folder
    if "chat_engine" not in st.session_state:
        with st.spinner("Ingesting..."):
            with open(f"data/{file.name}", "wb") as f:
                f.write(file.getbuffer())

            documents = SimpleDirectoryReader(
                input_files=[f"./data/{file.name}"]).load_data()

            nodes = pipeline.run(documents=documents)

            index = VectorStoreIndex(nodes)
            memory = ChatMemoryBuffer.from_defaults(token_limit=32768)

            chat_engine = index.as_chat_engine(
                memory=memory,
                chat_mode="condense_plus_context",
                context_prompt=(
                    "You are an AI assistant who has access to the following documents: "
                    "If you don't know the answer, just say you don't know. DO NOT try to make up an answer."
                    "If the user has no documents, just say 'No relevant documents found'."
                    "Do not try to make up an answer."
                    "Do not reference the datasource."
                    "Here are the relevant documents for the context:\n"
                    "{context_str}"
                    "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
                )
            )

            st.session_state["chat_engine"] = chat_engine


for message in st.session_state.messages:
    # skip system message
    if message['role'] == 'system':
        continue

    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt := st.chat_input("What is up"):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    response = generate_response(prompt)
    st.session_state.messages.append(
        {"role": "assistant", "content": response})
