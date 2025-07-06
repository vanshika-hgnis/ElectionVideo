import ffmpeg
import subprocess
import os

# Set full path to ffmpeg.exe explicitly
ffmpeg_path = r"D:/C-Drive/ffmpeg-7.1.1-full_build/bin/ffmpeg.exe"

# Paths
input_video = "input/input.mp4"
split_1 = "part1.mp4"
split_2 = "part2.mp4"
merged_output = "merged_output.mp4"

# Split time in seconds
split_time = 2

# --- STEP 1: Split Video ---
def split_video():
    # First part
    cmd1 = ffmpeg.input(input_video, ss=0, t=split_time)\
        .output(split_1, c='copy')\
        .compile(cmd=ffmpeg_path)
    subprocess.run(cmd1)

    # Second part
    cmd2 = ffmpeg.input(input_video, ss=split_time)\
        .output(split_2, c='copy')\
        .compile(cmd=ffmpeg_path)
    subprocess.run(cmd2)

# --- STEP 2: Merge Video ---
def merge_videos():
    with open("files.txt", "w") as f:
        f.write(f"file '{split_1}'\n")
        f.write(f"file '{split_2}'\n")

    merge_cmd = [
        ffmpeg_path,
        "-f", "concat",
        "-safe", "0",
        "-i", "files.txt",
        "-c", "copy",
        merged_output
    ]
    subprocess.run(merge_cmd)

# --- MAIN ---
if __name__ == "__main__":
    print("🔧 Splitting video...")
    split_video()
    # print("🔧 Merging video...")
    # merge_videos()
    # print(f"✅ Done! Output saved as: {merged_output}")
