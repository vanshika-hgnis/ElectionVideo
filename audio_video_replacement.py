import os
import asyncio
import edge_tts
from pydub import AudioSegment

# Paths
original_audio = "temp_audio.aac"
replacement_text = "वंशिका जी नमस्कार"
tts_audio_raw = "tts_audio.raw"  # Edge output
tts_audio_mp3 = "tts_audio.mp3"  # convertable
tts_audio_wav = "tts_audio.wav"
final_audio = "modified_audio.m4a"


# Step 1: Generate Hindi voice using Edge TTS
async def generate_tts():
    communicate = edge_tts.Communicate(
        text=replacement_text,
        voice="hi-IN-MadhurNeural",
        rate="+0%",
    )
    await communicate.save(tts_audio_mp3)  # Save as MP3


asyncio.run(generate_tts())

# ✅ Step 2: Convert to WAV using ffmpeg
os.system(f"ffmpeg -y -i {tts_audio_mp3} {tts_audio_wav}")

# ✅ Step 3: Load audio
original = AudioSegment.from_file(original_audio)
replacement = AudioSegment.from_wav(tts_audio_wav)

# Optional: Pad or trim TTS to 1s
replacement = replacement.set_frame_rate(44100)
replacement_duration = len(replacement)
print(f"🔊 Replacement duration: {replacement_duration} ms")

# Replace audio at 1s mark
start_ms = 1000
end_ms = start_ms + replacement_duration

before = original[:start_ms]
after = original[end_ms:]
final = before + replacement + after

# ✅ Step 4: Export final audio (M4A or AAC)
final.export(final_audio, format="ipod")  # M4A
# Or for AAC format:
# final.export("modified_audio.aac", format="adts")

print("✅ Audio replaced and saved at:", final_audio)
