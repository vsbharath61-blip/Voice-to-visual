from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import whisper
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import uuid
import os

app = FastAPI()

# Load Whisper model
whisper_model = whisper.load_model("base")

# Load Stable Diffusion (CPU safe mode)
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)
pipe = pipe.to(device)

@app.post("/generate")
async def generate(audio: UploadFile = File(...)):
    
    # Save uploaded audio
    temp_audio = f"temp_{audio.filename}"
    with open(temp_audio, "wb") as f:
        f.write(await audio.read())

    # Convert voice to text
    result = whisper_model.transcribe(temp_audio)
    prompt = result["text"]

    # Generate image
    image = pipe(prompt).images[0]

    # Save image
    image_name = f"{uuid.uuid4()}.png"
    image.save(image_name)

    # Remove temp audio
    os.remove(temp_audio)

    return JSONResponse({
        "recognized_text": prompt,
        "image_file": image_name
    })