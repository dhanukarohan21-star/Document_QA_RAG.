import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="Document QA Assistant")
st.title("AI Document Question Answering System")
st.write("Upload documents and ask questions about their contents.")

api_key = os.getenv("OPENAI_API_KEY")

uploaded_files = st.file_uploader(
	"Upload PDF or TXT documents",
	type=["pdf", "txt"],
	accept_multiple_files=True
)

top_k = st.slider("Retrieved passages", 1, 8, 4)


@st.cache_resource
def get_embedding_model():
	return HuggingFaceEmbedding(
		model_name="sentence-transformers/all-MiniLM-L6-v2"
	)


if st.button("Build document index"):
	if not uploaded_files:
		st.error("Upload at least one PDF or TXT file.")
	else:
		try:
			with st.spinner("Reading and indexing documents..."):
				with tempfile.TemporaryDirectory() as folder:
					for file in uploaded_files:
						name = Path(file.name).name
						Path(folder, name).write_bytes(file.getvalue())

					Settings.embed_model = get_embedding_model()

					docs = SimpleDirectoryReader(
						input_dir=folder,
						required_exts=[".pdf", ".txt"]
					).load_data()

					if not docs:
						raise ValueError("No readable document text found.")

					st.session_state["index"] = VectorStoreIndex.from_documents(
						docs,
						show_progress=True
					)

			st.success("Documents indexed successfully!")

		except Exception as e:
			st.error(f"Indexing failed: {e}")

if "index" in st.session_state:
	question = st.text_input("Enter your question")

	if st.button("Get answer"):
		if not api_key:
			st.error("Configure OPENAI_API_KEY in your .env file.")
		elif not question.strip():
			st.warning("Please enter a question.")
		else:
			try:
				with st.spinner("Searching and generating answer..."):
					nodes = st.session_state["index"].as_retriever(
						similarity_top_k=top_k
					).retrieve(question)

					if not nodes:
						st.info("No relevant passages were found.")
					else:
						context = "\n\n".join(
							f"[Source {i}] "
							f"{item.node.metadata.get('file_name', 'Document')}\n"
							f"{item.node.get_content()}"
							for i, item in enumerate(nodes, start=1)
						)

						prompt = ChatPromptTemplate.from_messages([
							(
								"system",
								"""Answer only from the supplied context.
If the answer is missing, say you could not find it.
Do not invent facts. Cite source labels when useful."""
							),
							(
								"human",
								"Context:\n{context}\n\nQuestion: {question}"
							)
						])

						chain = (
							prompt
							| ChatOpenAI(
								model="gpt-4o-mini",
								temperature=0,
								api_key=api_key
							)
							| StrOutputParser()
						)

						answer = chain.invoke({
							"context": context,
							"question": question
						})

						st.subheader("Answer")
						st.write(answer)

						st.subheader("Retrieved sources")
						for i, item in enumerate(nodes, start=1):
							with st.expander(f"Source {i}"):
								st.write(item.node.metadata)
								st.write(item.node.get_content())

			except Exception as e:
				st.error(f"Question answering failed: {e}")
