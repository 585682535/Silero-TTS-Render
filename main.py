from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
import torchaudio
import base64
import io

# Загрузка модели
model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language='ru',
                                     speaker='baya')

app = FastAPI()

class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
def tts(req: TTSRequest):
    audio = model.apply_tts(text=req.text)
    buffer = io.BytesIO()
    torchaudio.save(buffer, torch.tensor([audio]), 16000, format="wav")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {"audioContent": encoded}
