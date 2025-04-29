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
from llama_cpp import Llama
import llama_cpp
from typing import Optional
from llama_cpp import LogitsProcessorList
from lmformatenforcer import CharacterLevelParser
from lmformatenforcer.integrations.llamacpp import build_llamacpp_logits_processor, build_token_enforcer_tokenizer_data
from lmformatenforcer import JsonSchemaParser

LlamaModel = Llama(model_path="YOUR_HOME_DIR/deepseek-model-gguf/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00011.gguf", n_threads=128, temp=0.6,  n_ctx=1024*32)
tokenizer_data = build_token_enforcer_tokenizer_data(LlamaModel)

def generate(prompt: str, schema: BaseModel):
    character_level_parser = JsonSchemaParser(schema.model_json_schema())
    logits_processors = LogitsProcessorList([build_llamacpp_logits_processor(tokenizer_data, character_level_parser)])
    response = LlamaModel(
        prompt,
        max_tokens=None,
        echo=False,
        logits_processor=logits_processors
    )
    text = response["choices"][0]["text"]
    print(text)
    json_result = json.loads(text)

    # Return valid JSON object according to the schema DeepEval supplied
    return schema(**json_result)

class Test(BaseModel):
    greeting: constr(max_length=30)
print(generate(prompt="Hi", schema=Test))


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

max_len_q = 2048
max_len_a = 4096
max_len_t = 8192
class QAPair(BaseModel):
    initialThoughts: constr(max_length=max_len_t)
    question: constr(max_length=max_len_q)
    answerThoughts: constr(max_length=max_len_t)
    isAnswerable: bool
    isDuplicate: bool
    answer: constr(max_length=max_len_a)

class QADataset(BaseModel):
    definitionPair: QAPair
    diagnosticworkupPair: QAPair
    caserelatedPair: QAPair
    unanswerablePair: QAPair

# Save Dataset
from datasets import Dataset
def saveDS(ds, savePath="YOUR_HOME_DIR/dataset_chapters_complex_deepseek_cpu"):
    print(f"New Dataset len: {len(ds)}")
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
        
        if len(baseText) < 1000:
            continue
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
        
        scheme = QADataset
        exampleStr = None
        if scheme is QADataset:
            exampleStr = '{"definitionPair": {"initialThoughts": str, "question": str, "answerThoughts": str, "isAnswerable": bool, "isDuplicate": bool, "answer": str}, "diagnosticworkupPair": {"initialThoughts": str, ...}, "caserelatedPair": {"initialThoughts": str, ...}, "unanswerablePair": {"initialThoughts": str, ...}}'
        elif scheme is QAPair:
            exampleStr = '{"initialThoughts": str, "question": str, "answerThoughts": str, "isAnswerable": bool, "isDuplicate": bool, "answer": str}'
        else:
            print("UNKOWN SCHEME")
        schemeStr = f"Structure your output as JSON in the following scheme:\n{scheme.model_json_schema()}\nIn readable json the format is:\n{exampleStr}"
        input_text = input_text.replace("!SCHEME!", schemeStr)
        
        for attempt in range(5):
            output = None
            try:
                print("Generating with prompt:")
                print(input_text)
                output = generate(prompt=input_text, schema=scheme)
                print("Done generating")
                break
            except ValueError as e:
                print(e)
                print(f"Errored at attempt {attempt}")
            
        if output != None:
            if type(output) is QAPair:
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
            elif type(output) is QADataset:
                entries = [output.definitionPair, output.diagnosticworkupPair, output.caserelatedPair, output.unanswerablePair]
                for entry in entries:
                    if entry.isAnswerable and not entry.isDuplicate:
                        history[i].append(entry)
                        data_collection.append(entry)
                    else:
                        print(f"Skipping entry {entry}")
                try:
                    saveDS(data_collection)
                except Exception as e:
                    print("Failed to save dataset!")
                    print(e)
            else:
                print(f"Unknown output type {output}")


