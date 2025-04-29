import gradio as gr
from docling.document_converter import DocumentConverter
import os
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import FigureElement, InputFormat, Table
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
import json
from pathlib import Path
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrMacOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
import asyncio
import sys

from docling_core.types import DoclingDocument
from datetime import datetime
import requests
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.vector_stores.milvus import MilvusVectorStore
from docling.document_converter import DocumentConverter
from llama_index.node_parser.docling import DoclingNodeParser
from docling.chunking import HybridChunker
from pathlib import Path
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.vector_stores.milvus import MilvusVectorStore
from docling.document_converter import DocumentConverter
from llama_index.node_parser.docling import DoclingNodeParser
from docling.chunking import HybridChunker

hf_token = "HUGGINGFACE TOKEN HERE"
os.environ["HUGGINGFACEHUB_API_TOKEN"]=hf_token
os.environ["HF_TOKEN"]=hf_token
os.environ['HF_HOME'] = 'YOUR_HOME_DIR/.cache/huggingface/'
os.environ['TRANSFOMERS_CACHE'] = 'YOUR_HOME_DIR/.cache/huggingface/'
import transformers
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

default_settings = {
    "ollama_endpoint": "http://leinevmgpu001.mh-hannover.local:11434",
    "base_path": "YOUR_HOME_DIR/gui/",
    "persist_dir": "YOUR_HOME_DIR/datasets/persistent_vector_store_gui",
    "database_path": "YOUR_HOME_DIR/datasets/docling_md_vectordb_gui.db"
}

EMBED_MODEL = HuggingFaceEmbedding(model_name="abhinand/MedEmbed-large-v0.1")

Settings.embed_model = EMBED_MODEL
embed_dim = len(EMBED_MODEL.get_text_embedding("hi"))

def request_model_names(settings):
    #hostname = "leinevmgpu001.mh-hannover.local"
    hostname = settings["ollama_endpoint"]
    url = f"{hostname}/api/tags"  # Adjust the URL if needed based on your server setup
    print(url)
    try:
        # Send a GET request to the server to fetch installed models
        response = requests.get(url)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            models = response.json()  # Assuming the server responds with a JSON list of models
            model_names = [m["name"] for m in models["models"]]
            print(model_names)
            return gr.Dropdown(choices=model_names, interactive=True)
        else:
            print(f"Failed to fetch models. HTTP Status Code: {response.status_code}")
            return ["Error"]
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return ["Error"]

def process_pdf(file, settings):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_name = Path(file.name).name
    
    output_path = f"{settings['base_path']}/{file_name}_{timestamp}.md"

    IMAGE_RESOLUTION_SCALE = 2.0

    source = file
    
    pipeline_options = PdfPipelineOptions(do_table_structure=True)
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE  # use more accurate TableFormer model
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.generate_table_images = True
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend)
        }
    )
    
    result = converter.convert(source)
    doc_filename = result.input.file.stem

    text = result.document.export_to_markdown()
    with open(output_path, "wt", encoding="utf-8") as f:
        f.write(text)
    
    return output_path

def create_vector_store(settings, markdown_file_path):
    print(markdown_file_path)
    SOURCE = markdown_file_path
    
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    reader = DoclingReader()
    node_parser = MarkdownNodeParser()
    chunker = HybridChunker()

    milvus_db = settings["database_path"]
    persistent_dir = settings["persist_dir"]
    
    vector_store = MilvusVectorStore(
        uri=str(Path(milvus_db)),
        dim=embed_dim,
        overwrite=True,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents=reader.load_data(SOURCE),
        transformations=[node_parser],
        storage_context=storage_context,
        embed_model=EMBED_MODEL,
    )
    
    storage_context.persist(persist_dir=persistent_dir)

    return f"Milvus DB: {milvus_db}\nPersistent Storage: {persistent_dir}"

def load_vector_store(settings):
    persist_dir = settings['persist_dir']

    database_path = settings['database_path']
    
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    vector_store = MilvusVectorStore(
        uri=str(Path(database_path)),
        dim=embed_dim,
        overwrite=False,
    )
    
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir=persist_dir),
        graph_store=SimpleGraphStore.from_persist_dir(persist_dir=persist_dir),
        index_store=SimpleIndexStore.from_persist_dir(persist_dir=persist_dir),
    )
    
    index = load_index_from_storage(storage_context)
    retriever = index.as_retriever(similarity_top_k=5)

    return vector_store, retriever

def retrieve_sources(question, retriever):
    retrieved_docs = retriever.retrieve(question)
    print(f"Retrieved {len(retrieved_docs)} docs")
    sources = [s.get_content(s.metadata) for s in retrieved_docs]
    sourcesStr = "\n\n".join(sources)

    return (sources, sourcesStr)

def generate_response(question, ollama_endpoint, model_name, sourcesTuple):

    sourcesStr = sourcesTuple[1]
    
    QUERY = f"### Input:\n{question}\nContext:\n{sourcesStr}\n"
    
    data = {
        "model": model_name,
        "prompt": QUERY,
        "stream": True,
        "options": {"num_predict":4096}
    }
    url = f"{ollama_endpoint}/api/generate"
    response = requests.post(url, json=data, timeout=120)
    con_text = ""
    for l in response.text.split("\n"):
        try:
            obj = json.loads(l)
        except:
            continue    
        if "done" in obj:
            if obj["done"] == True:
                yield con_text
            else:
                con_text += obj["response"]
                yield con_text

def infer_with_ollama(prompt, settings, model_name):
    print(model_name)
    vector_store, retriever = load_vector_store(settings)

    sourcesTup = retrieve_sources(prompt, retriever)

    ollama_endpoint = settings['ollama_endpoint']

    for s in generate_response(prompt, ollama_endpoint, model_name, sourcesTup):
        yield s, sourcesTup[1]

    vector_store.client.close()

def save_settings(ollama_endpoint, basePath, persistent_dir, milvus_db):
    base_path_new = basePath if not basePath.endswith("/") else basePath[:-1]
    return {"ollama_endpoint": ollama_endpoint, "base_path": base_path_new, "persist_dir": persistent_dir, "database_path": milvus_db}

def use_settings_example(file, settings):
    return f"Using settings: {settings}\nReceived: {file.name}"

with gr.Blocks() as settings_tab:
    settings_state = gr.State(default_settings)
    gr.Markdown("### Settings")
    ollama_endpoint = gr.Textbox(label="Ollama Endpoint", value=default_settings["ollama_endpoint"])
    basePath = gr.Textbox(label="Base Path", value=default_settings["base_path"])
    persistent_dir = gr.Textbox(label="[RAG] Persistent Storage Directory", value=default_settings["persist_dir"])
    milvus_db = gr.Textbox(label="[RAG] Milvus Database Path", value=default_settings["database_path"])
    
    save_button = gr.Button("Save Settings")
    settings_output = gr.Textbox(label="Current Settings")
    
    save_button.click(
        fn=save_settings,
        inputs=[ollama_endpoint, basePath, persistent_dir, milvus_db],
        outputs=[settings_state],
    )
    save_button.click(
        fn=lambda x: str(x),
        inputs=settings_state,
        outputs=settings_output
    )

# Tab 1: Document Conversion
with gr.Blocks() as doc_conversion_tab:
    gr.Markdown("### Document Conversion")
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
    convert_btn = gr.Button("Convert Document")
    pdf_output = gr.Textbox(label="Save Path")

    make_rag_btn = gr.Button("Prepare Document for RAG")
    rag_output = gr.Textbox(label="Prepare RAG Output")

    convert_btn.click(fn=process_pdf, inputs=[pdf_input, settings_state], outputs=pdf_output)
    make_rag_btn.click(fn=create_vector_store, inputs=[settings_state, pdf_output], outputs=rag_output)

# Tab 2: LLM Chat Interface
with gr.Blocks() as llm_chat_tab:
    gr.Markdown("### LLM Chat Interface")
    with gr.Row():
        models_dropdown = gr.Dropdown(['None'], allow_custom_value=True)
        load_models_btn = gr.Button("Refresh")

        load_models_btn.click(fn=request_model_names, inputs=[settings_state], outputs=models_dropdown)
    with gr.Row():
        user_prompt = gr.Textbox(label="Enter your prompt")
    chat_btn = gr.Button("Send to LLM")

    with gr.Row():   
        chat_output = gr.Textbox(label="LLM Response", lines=10, max_lines=20)
        chat_sources = gr.Textbox(label="Sources", lines=10, max_lines=20)

    chat_btn.click(fn=infer_with_ollama, inputs=[user_prompt, settings_state, models_dropdown], outputs=[chat_output, chat_sources])

# Combine into Tabs
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Settings"):
            settings_tab.render()
        with gr.TabItem("Document Conversion"):
            doc_conversion_tab.render()
        with gr.TabItem("LLM Chat"):
            llm_chat_tab.render()


hostname = "leinevmgpu004.mh-hannover.local"

if len(sys.argv) > 1:
    hostname = sys.argv[1]

demo.launch(server_name=hostname)


