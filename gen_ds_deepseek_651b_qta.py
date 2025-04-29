import os
hf_token = "HUGGINGFACE TOKEN HERE"
os.environ["HUGGINGFACEHUB_API_TOKEN"]=hf_token
os.environ["HF_TOKEN"]=hf_token
os.environ['HF_HOME'] = 'YOUR_HOME_DIR/.cache/huggingface/'
os.environ['TRANSFOMERS_CACHE'] = 'YOUR_HOME_DIR/.cache/huggingface/'
os.environ['LMFE_STRICT_JSON_FIELD_ORDER'] = "True"

#Wait for ollama server starting
import time
time.sleep(5)

import json
from pydantic import BaseModel, conlist, constr
from docling_core.types import DoclingDocument
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.file import FlatReader
from pathlib import Path
import requests
import ollama

def generate(prompt: str, schema: BaseModel):
    data = {
        "model": "deepseek-r1:671b",
        "prompt": prompt,
        "stream": False,
        "format": schema.model_json_schema()
    }
    url = "http://172.24.120.177:11434/api/generate"
    response = requests.post(url, json=data, timeout=60*60*1000)
    text = response.json()["response"]
    print(text)
    json_result = json.loads(text)

    # Return valid JSON object according to the schema DeepEval supplied
    return schema(**json_result)

class Test(BaseModel):
    greeting: constr(max_length=30)
generate(prompt="Hi", schema=Test)


SOURCE = r"YOUR_HOME_DIR/guideline_edit.md"

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

max_len_q = 1024
max_len_a = 2048
#max_len_t = 2048
class QAPair(BaseModel):
    initialThoughts: constr(pattern=r"^<think>.*", min_length=7)
    question: constr(min_length=1, max_length=max_len_q)
    answerThoughts: constr(pattern=r"^<think>.*", min_length=7)
    isAnswerable: bool
    isDuplicate: bool
    answer: constr(min_length=1)
    
# Save Dataset
from datasets import Dataset
def saveDS(ds, savePath="YOUR_HOME_DIR/dataset_chapters_complex_deepseek"):
    print(f"Dataset len: {len(ds)}")
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

#class QADataset(BaseModel):
#    data: conlist(QAPair, min_length=10, max_length=20)
#    generatedSufficientPairs: bool
history = {}
for iteration in range(30):
    for i, chapter in enumerate(chapters):
        baseText = chapter["text"]
        #print(f"=== {i} ===")
        #print(f"chunk length: {len(baseText)}")
        #print(f"chunk.text:\n{repr(f'{baseText[:100]}…')}")
        
        if not i in history.keys():
            history[i] = []

        context_txt = baseText
        print(f"Generating questions for chapter {i}")
        input_text = None
        with open("YOUR_HOME_DIR/prompt_with_examples_no_ref_context.txt", "rt", encoding="utf-8") as f:
            input_text = f.read()
        
        input_text = input_text.replace("!CONTEXT!", context_txt)
        
        history_txt = ""
        if len(history[i]) > 0:
            history_header = "The following section contains all the previously generated questions for this context. Make sure to not generate new questions that are similar to those presented next section!\n### History:\n"
            q_history = "\n".join([f"Q{index}: {o.question}" for index, o in enumerate(history[i])])
            history_txt = f"{history_header}\n{q_history}"
        input_text = input_text.replace("!HISTORY!", history_txt)
        
        scheme = QAPair
        for attempt in range(5):
            output = None
            try:
                output = generate(prompt=input_text, schema=scheme)
                break
            except ValueError as e:
                print(f"Errored at attempt {attempt}")
            
        if output != None:
            if output.isAnswerable and not output.isDuplicate:
                history[i].append(output)
                data_collection.append(output)
            else:
                print(f"Skipping entry {output}")
            try:
                saveDS(data_collection)
            except Exception as e:
                print("Failed to save dataset!")
                print(e)


