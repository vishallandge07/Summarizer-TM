from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
import torch
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Text Sumarrizer", description="fast summarization", version="1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model.to(device)


templates = Jinja2Templates(directory="templates")

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return text

def summarize_dialogue(dialogue : str):
    # clean
    dialogue = clean_data(dialogue)

    #tokenize
    inputs = tokenizer(
        dialogue,
        max_length=512,
        padding = "max_length",
        truncation = True,
        return_tensors = "pt"
    ).to(device)
    
      # Summary  Generation
    model.to(device)

    targets = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        max_length = 150,
        num_beams = 4,
        early_stopping = True 
    )


    #token_ids to summary => decoding
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

@app.post("/summarize/")
async def summarize(dialogue_intput: DialogueInput):
    summary = summarize_dialogue(dialogue_intput.dialogue)
    return {"summary":summary}

