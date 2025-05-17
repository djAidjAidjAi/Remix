import torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration


processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

inputs = processor(
    text=["Acoustic sounds, soft melodies and delicate vocals provide listeners with comfort and deep emotional empathy. It gives the impression that you are listening to a series of emotional essays or OSTs of a calm romance drama, which are good to listen to when you want to be immersed in thoughts or emotions like late at night, dawn, or autumn. It is the perfect playlist when you want to get out of a noisy daily life and calm down."],
    padding=True,
    return_tensors="pt",
)

audio_values = model.generate(**inputs, max_new_tokens=256)


# 샘플레이트 (MusicGen은 32kHz 고정)
sample_rate = 32000

waveform = audio_values[0]  # shape: (1, 1, N)
waveform = waveform.squeeze()  # shape: (N,)
waveform = waveform.unsqueeze(0)  # shape: (1, N) → mono channel

torchaudio.save("output.wav", waveform.cpu(), sample_rate)