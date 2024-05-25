import cv2
import ffmpeg

def detect_face_coordinates(video_path):
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    face_coords = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_coords = (x, y, w, h)
            break
    
    cap.release()
    return face_coords

def crop_video(input_path, output_path, face_coords):
    x, y, w, h = face_coords
    crop_size = min(w, h)  # Make it a square crop
    
    (
        ffmpeg
        .input(input_path)
        .crop(x, y, crop_size, crop_size)
        .filter('scale', 224, 224)
        .filter('fps', fps=17)
        .output(output_path)
        .run()
    )

def main():
    video_path = 'obama-debt.mp4'
    output_path = 'output_cropped.mp4'
    
    face_coords = detect_face_coordinates(video_path)
    if face_coords:
        print(f"Detected face coordinates: {face_coords}")
        crop_video(video_path, output_path, face_coords)
        print(f"Cropped video saved as {output_path}")
    else:
        print("No face detected in the video.")

if __name__ == "__main__":
    main()
