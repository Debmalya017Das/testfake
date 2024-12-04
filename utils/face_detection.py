# import os
# import face_recognition
# from PIL import Image

# def extract_faces_from_frames(input_folder, output_folder):
#     """Detects and extracts faces from frames."""
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     for file_name in os.listdir(input_folder):
#         frame_path = os.path.join(input_folder, file_name)
#         frame = face_recognition.load_image_file(frame_path)
#         face_locations = face_recognition.face_locations(frame)

#         for i, face_location in enumerate(face_locations):
#             top, right, bottom, left = face_location
#             face_image = frame[top:bottom, left:right]
#             face_pil = Image.fromarray(face_image)
#             face_pil.save(os.path.join(output_folder, f"{file_name}face{i}.jpg"))

#     print(f"Faces extracted to {output_folder}")

import cv2
import os

def extract_faces_from_frames(input_folder, output_folder):
    """Detects and extracts faces from frames using OpenCV."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    for file_name in os.listdir(input_folder):
        frame_path = os.path.join(input_folder, file_name)
        frame = cv2.imread(frame_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Extract and save each face
        for i, (x, y, w, h) in enumerate(faces):
            face = frame[y:y+h, x:x+w]
            output_path = os.path.join(output_folder, f"{file_name}_face{i}.jpg")
            cv2.imwrite(output_path, face)
    
    print(f"Faces extracted to {output_folder}")