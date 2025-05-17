from audiocraft.models import MusicGen
import torchaudio
import torch

# 1. 모델 로드
model = MusicGen.get_pretrained("facebook/musicgen-melody")  # melody remix용
model.set_generation_params(duration=10)  # 생성 길이 (초)

# 2. 로컬 오디오 로딩
waveform, sr = torchaudio.load("my_song.wav")

# 3. 리샘플링 (필수: MusicGen은 32000Hz만 지원)
if sr != 32000:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=32000)
    waveform = resampler(waveform)
    sr = 32000

# 4. 배치 형식으로 맞추기 (batch_size=2 권장, model 내부가 그 구조를 기대함)
waveform = waveform.unsqueeze(0).repeat(2, 1, 1)  # shape: (2, 1, N)

# 5. 텍스트 스타일 프롬프트 입력
descriptions = [
    "dreamy lo-fi remix with soft keys and ambient textures",
    "nostalgic chillhop beat with mellow guitar and tape hiss"
]

# 6. 음악 생성 (멜로디 + 텍스트 기반)
output = model.generate_with_chroma(
    descriptions=descriptions,
    melody_wavs=waveform,
    melody_sample_rate=sr,
    progress=True,
    return_tokens=False
)

# 7. 결과 저장 (첫 번째 트랙만 저장)
torchaudio.save("remix.wav", output[0].cpu(), 32000)