import os
import torch
import torchaudio
import argparse
import look2hear.models
import warnings
from pydub import AudioSegment

SEGMENT_SECONDS = 10  # length of each segment in seconds
OVERLAP_SECONDS = 1   # overlap to ensure no gaps (kan justeras)

# Suppress Torchaudio backend warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torchaudio._backend.utils"
)

# Load audio, auto-convert MP3/m4a/aiff to WAV
def load_audio(file_path, device="cuda"):
    ext = os.path.splitext(file_path)[1].lower()
    temp_file = None
    if ext in ['.mp3', '.m4a', '.aiff', '.aif']:
        temp_file = file_path + ".tmp.wav"
        AudioSegment.from_file(file_path).export(temp_file, format="wav")
        file_path = temp_file

    audio, samplerate = torchaudio.load(file_path)
    audio = audio.unsqueeze(0)  # [1, 1, samples]
    audio = audio.to(device)

    return audio, samplerate, temp_file  # return temp file path for cleanup

def save_audio(file_path, audio, samplerate=44100):
    audio = audio.squeeze(0).cpu()
    torchaudio.save(file_path, audio, samplerate)

def process_segments(model, audio, samplerate, overlap=OVERLAP_SECONDS):
    segment_length = SEGMENT_SECONDS * samplerate
    overlap_length = int(overlap * samplerate)
    hop_length = segment_length - overlap_length
    total_samples = audio.shape[-1]
    output_chunks = []
    cursor = 0

    with torch.no_grad():
        while cursor < total_samples:
            end = min(cursor + segment_length, total_samples)
            segment = audio[:, :, cursor:end]

            # Pad sista segmentet om det är för kort
            if end - cursor < segment_length:
                pad_size = segment_length - (end - cursor)
                segment = torch.nn.functional.pad(segment, (0, pad_size))

            out = model(segment)

            # Trimma bort paddingen på sista segmentet
            if end - cursor < segment_length:
                out = out[:, :, :end - cursor]

            # Hantera överlappning
            if cursor == 0:
                # Första segmentet: ta allt utom sista overlappen
                output_chunks.append(out[:, :, :-overlap_length])
            elif end >= total_samples:
                # Sista segmentet: ta allt
                output_chunks.append(out)
            else:
                # Mellansegment: ta allt utom sista overlappen
                output_chunks.append(out[:, :, :-overlap_length])

            cursor += hop_length

    return torch.cat(output_chunks, dim=-1)

def main(input_file, output_file):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pretrained model
    model_file = r"Apollo\pytorch_model.bin"
    model = look2hear.models.BaseModel.from_pretrain(
        model_file,
        sr=44100,
        win=20,
        feature_dim=256,
        layer=6
    ).to(device)
    model.eval()

    # Load audio
    audio, samplerate, temp_file = load_audio(input_file, device=device)
    # Process audio with overlap-and-crop
    output_audio = process_segments(model, audio, samplerate)
    # Save output
    save_audio(output_file, output_audio, samplerate)

    # Delete temporary WAV file if created
    if temp_file is not None and os.path.exists(temp_file):
        os.remove(temp_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Inference Script")
    parser.add_argument("--in_wav", type=str, required=True, help="Path to input wav/mp3 file")
    parser.add_argument("--out_wav", type=str, required=True, help="Path to output wav file")
    args = parser.parse_args()

    main(args.in_wav, args.out_wav)
