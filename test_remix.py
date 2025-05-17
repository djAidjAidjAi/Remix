from transformers import AutoProcessor, MusicgenForConditionalGeneration
from datasets import load_dataset
import torchaudio

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

dataset = load_dataset("sanchit-gandhi/gtzan", split="train", streaming=True)
sample = next(iter(dataset))["audio"]

# take the first half of the audio sample
sample["array"] = sample["array"][: len(sample["array"]) // 2]

inputs = processor(
    audio=sample["array"],
    sampling_rate=sample["sampling_rate"],
    text=["80s blues track with groovy saxophone"],
    padding=True,
    return_tensors="pt",
)
audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)



# 샘플레이트 (MusicGen은 32kHz 고정)
sample_rate = 32000

waveform = audio_values[0]  # shape: (1, 1, N)
waveform = waveform.squeeze()  # shape: (N,)
waveform = waveform.unsqueeze(0)  # shape: (1, N) → mono channel

torchaudio.save("output_remix.wav", waveform.cpu(), sample_rate)