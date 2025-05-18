from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.responses import FileResponse
from audiocraft.models import MusicGen
import torchaudio
import torch
import os
import asyncio

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = MusicGen.get_pretrained("facebook/musicgen-melody")
model.set_generation_params(duration=10)

@app.post("/generate")
async def generate_music(
    audio_file: UploadFile = File(...),
    prompt1: str = Form(...),
    prompt2: str = Form(...)
):
    # Save uploaded audio
    input_path = "input.wav"
    output_path = "output.wav"
    with open(input_path, "wb") as f:
        f.write(await audio_file.read())

    # Load and resample
    waveform, sr = torchaudio.load(input_path)
    if sr != 32000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=32000)(waveform)
    waveform = waveform.unsqueeze(0).repeat(2, 1, 1)

    # Generate
    output = model.generate_with_chroma(
        descriptions=[prompt1, prompt2],
        melody_wavs=waveform,
        melody_sample_rate=32000,
        progress=True
    )

    torchaudio.save(output_path, output[0].cpu(), 32000)
    return {"message": "Success", "file": output_path}

@app.get("/download")
def download_audio():
    return FileResponse("output.wav", media_type="audio/wav", filename="remix.wav")

# -----------------------
# 🎭 MOCK ENDPOINTS BELOW
# -----------------------

@app.post("/mock_generate")
async def mock_generate():
    await asyncio.sleep(5)  # simulate processing delay
    return JSONResponse(content={"message": "Mock Success", "file": "empty.mp3"})

@app.get("/mock_download")
def mock_download():
    return FileResponse("empty.mp3", media_type="audio/mpeg", filename="mock_remix.mp3")
    return FileResponse(output_path, media_type="audio/wav", filename="output.wav")
    # return {"message": "Success", "file": output_path}
