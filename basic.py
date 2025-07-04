# from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip, concatenate_videoclips
from gtts import gTTS
import os
from moviepy  import * 

# ---- Input Configuration ----
VIDEO_PATH = "input/video.mp4"
OUTPUT_PATH = "output/final_video.mp4"

# Timestamp in seconds where word needs to be replaced
TARGET_TIMESTAMP = 5.0  # e.g., 5 seconds
MUTE_DURATION = 2.0  # seconds to mute before inserting new audio

OLD_TEXT = "नमस्ते"
NEW_TEXT = "प्रणाम"
TEXT_POSITION = ('center', 'bottom')  # or (x, y) like (200, 300)

# ---- Generate replacement audio ----
tts = gTTS(NEW_TEXT, lang='hi')
tts.save("replace_audio.mp3")
replacement_audio = AudioFileClip("replace_audio.mp3")

# ---- Load original video ----
clip = VideoFileClip(VIDEO_PATH)

# ---- Step 1: Cut the original video into 3 parts ----
before = clip.subclip(0, TARGET_TIMESTAMP)
after = clip.subclip(TARGET_TIMESTAMP + MUTE_DURATION, clip.duration)

# ---- Step 2: Create a muted section ----
muted = clip.subclip(TARGET_TIMESTAMP, TARGET_TIMESTAMP + MUTE_DURATION).without_audio()

# ---- Step 3: Overlay new audio ----
muted = muted.set_audio(replacement_audio.set_duration(muted.duration))

# ---- Step 4: Add new text overlay ----
text_overlay = TextClip(NEW_TEXT, fontsize=60, color='white', font='Arial-Bold')
text_overlay = text_overlay.set_duration(muted.duration).set_position(TEXT_POSITION)
muted = CompositeVideoClip([muted, text_overlay])

# ---- Step 5: Combine all clips ----
final = concatenate_videoclips([before, muted, after])
final.write_videofile(OUTPUT_PATH, codec="libx264", audio_codec="aac")

# ---- Cleanup ----
os.remove("replace_audio.mp3")
