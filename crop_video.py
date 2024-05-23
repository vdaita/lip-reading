from moviepy.editor import VideoFileClip

def trim_and_crop(input_video, output_video="output.mp4"):
  """Trims and crops a video, reducing the frame rate to 15fps.

  Args:
    input_video: Path to the input video file.
    output_video: Path to the output video file.
  """

  # Load the video
  video = VideoFileClip(input_video)

  # Trim the video
  video = video.subclip(45, video.duration - 15)

  # Crop the video
  video = video.crop(x1=546, y1=180, x2=746, y2=380)

  # Reduce the frame rate
  video = video.set_fps(15)

  # Save the edited video
  video.write_videofile(output_video)

# Example usage:
input_video_path = "obama-debt.mp4"  # Replace with your input video file path
trim_and_crop(input_video_path)
