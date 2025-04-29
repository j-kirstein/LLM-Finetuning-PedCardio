import os
hf_token = "HUGGINGFACE TOKEN HERE"
os.environ["HUGGINGFACEHUB_API_TOKEN"]=hf_token
os.environ["HF_TOKEN"]=hf_token
os.environ['HF_HOME'] = 'YOUR_HOME_DIR/.cache/huggingface/'
os.environ['TRANSFOMERS_CACHE'] = 'YOUR_HOME_DIR/.cache/huggingface/'
os.environ['LMFE_STRICT_JSON_FIELD_ORDER'] = "True"

from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import json
import transformers
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel, conlist, constr
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (build_transformers_prefix_allowed_tokens_fn)
import re
import torch
from deepeval.models.base_model import DeepEvalBaseLLM, DeepEvalBaseEmbeddingModel
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import FiltrationConfig, ContextConstructionConfig
from transformers import BitsAndBytesConfig, AwqConfig
from docling.chunking import HybridChunker
from awq import AutoAWQForCausalLM

def pydantic_to_str(model: BaseModel) -> str:
    return str({field: (str if isinstance(model.model_fields[field].annotation, type) and issubclass(model.model_fields[field].annotation, str) else model.model_fields[field].annotation)
                for field in model.model_fields})

class SentenceFormerEmbedding(DeepEvalBaseEmbeddingModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = HuggingFaceEmbedding(model_name=self.model_name, device="cuda:0")

    def load_model(self):
        return self.model
    
    def embed_text(self, text: str):
        return self.model.get_text_embedding(text)

    async def a_embed_text(self, text: str):
        return self.embed_text(text)

    def embed_texts(self, texts: list):
        return self.model._get_text_embeddings(texts)

    async def a_embed_texts(self, texts: list):
        return self.embed_texts(texts)

    def get_model_name(self):
        return self.model_name
    
class Llama3_1_JSON(DeepEvalBaseLLM):
    def __init__(self):
        quantization_config = AwqConfig(
            bits=4,
            fuse_max_seq_len=512, # Note: Update this as per your use-case
            do_fuse=True,
        )
        
        model_id = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"

        model = AutoModelForCausalLM.from_pretrained(
          model_id,
          torch_dtype=torch.float16,
          low_cpu_mem_usage=True,
          device_map="auto",
          quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        model = self.load_model()
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_new_tokens=8192,
            #max_length=16384,
            do_sample=True,
            temperature=0.7,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Create parser required for JSON confinement using lmformatenforcer
        parser = JsonSchemaParser(schema.model_json_schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(
            pipeline.tokenizer, parser
        )

        # Output and load valid JSON
        output_dict = pipeline(prompt, prefix_allowed_tokens_fn=prefix_function
                              )
        output = output_dict[0]["generated_text"][len(prompt) :]
        print("Output:", output)
        print("\n\n")
        json_result = json.loads(output)

        # Return valid JSON object according to the schema DeepEval supplied
        return schema(**json_result)

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "LLama 3.1 JSON"
    
# INIT MODELS
#embedding_model = SentenceFormerEmbedding(model_name="abhinand/MedEmbed-large-v0.1")
llm = Llama3_1_JSON()

# Save Dataset
from datasets import Dataset
def saveDS(ds, savePath="YOUR_HOME_DIR/dataset_questions_only"):
    def gen():
        for d in ds:
            if type(d) is list:
                for e in d:
                    yield e.model_dump()
            elif type(d) is QAPair:
                yield d.model_dump()
            elif type(d) is dict:
                yield d
            else:
                print("unkown type!", type(d))
    
    # Create a Hugging Face dataset
    hf_dataset = Dataset.from_generator(gen)
    hf_dataset.save_to_disk(savePath)

# FULL DOCUMENT JSON SCHEME INFERENCE
from docling_core.types import DoclingDocument
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.file import FlatReader
from pathlib import Path

SOURCE = r"YOUR_HOME_DIR/guideline_edit.md"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

text_md = ""

data_collection = []


parser = MarkdownNodeParser()
md_docs = FlatReader().load_data(Path(SOURCE))
nodes = parser.get_nodes_from_documents(md_docs)

chapterStr = "# "
sectionStr = "## "
chapters = []
sections = []
tmp = ""
for n in nodes:
    text = n.text
    if text.startswith(chapterStr):
        if len(tmp) > 0:
            chapters.append({"text": tmp, "sections": sections})
        tmp = text
        sections = []
    else:
        tmp += "\n\n" + text

        if text.startswith(sectionStr):
            splits = text.split("\n")
            try:
                firstSplit = splits[0]
            except:
                firstSplit = "all"
            sections.append(firstSplit)
chapters.append({"text": tmp, "sections": sections})
for c in chapters:
    print(len(c["text"]))
    print("----------------")
print("chapter count:", len(chapters))

max_len_t = 2048
max_len_q = 512
class QAPair(BaseModel):
    thoughts: constr(min_length=1, max_length=max_len_t)
    question: constr(min_length=1, max_length=max_len_q)
    isDuplicate: bool

#class QADataset(BaseModel):
#    data: conlist(QAPair, min_length=10, max_length=20)
#    generatedSufficientPairs: bool

for i, chapter in enumerate(chapters):
    #Skip first section as it does not contain any information
    if i == 0:
        continue
    
    baseText = chapter["text"]
    print(f"=== {i} ===")
    print(f"chunk length: {len(baseText)}")
    print(f"chunk.text:\n{repr(f'{baseText[:100]}…')}")

    context_txt = baseText
    print(f"Generating questions for chapter {i}")
    out = []
    while len(out) < 30:
        input_text = None
        with open("YOUR_HOME_DIR/prompt_gen_questions.txt", "rt", encoding="utf-8") as f:
            input_text = f.read()
        
        input_text = input_text.replace("!CONTEXT!", context_txt)
        
        history_txt = ""
        if len(out) > 0:
            history_header = "The following section contains all the previously generated questions for this context. Make sure to not generate new questions that are similar to those presented next (History section)!\n### History:\n"
            q_history = "\n".join([f"Q{index}: {o.question}" for index, o in enumerate(out)])
            history_txt = f"{history_header}\n{q_history}"
        input_text = input_text.replace("!HISTORY!", history_txt)
        
        scheme = QAPair
        schemeStr = pydantic_to_str(scheme)
        input_text = input_text.replace("!SCHEME!", schemeStr)
        #print(input_text)
        output = None
        for attempt in range(5):
            try:
                output = llm.generate(prompt=input_text, schema=scheme)
                break
            except ValueError as e:
                print(f"Errored at attempt {attempt}")
            
        if output != None:
            if not output.isDuplicate:
                out.append(output)
            else:
                print(f"Skipping entry {output}")
        #print(out)

    print(f"Generated {len(out)} questions in total for the chapter {i}")
    data_collection.append(out)
    print("--------------------------------------")
    try:
        saveDS(data_collection)
    except Exception as e:
        print("Failed to save dataset!")
        print(e)


