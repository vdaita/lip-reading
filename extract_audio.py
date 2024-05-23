from moviepy.editor import VideoFileClip

def mp4_to_mp3(input_video, output_audio="output.mp3"):
  """Converts an MP4 video to MP3 audio.

  Args:
    input_video: Path to the input MP4 video file.
    output_audio: Path to the output MP3 audio file.
  """

  # Load the video
  video = VideoFileClip(input_video)

  # Extract the audio
  audio = video.audio

  # Save the audio as MP3
  audio.write_audiofile(output_audio)

# Example usage:
input_video_path = "obama-debt-cropped.mp4"  # Replace with your input video file path
mp4_to_mp3(input_video_path)
