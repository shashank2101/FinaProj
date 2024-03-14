import os
import face_recognition
import numpy as np
# Function to extract face encodings from an image
def extract_face_encodings(image_path):
    img = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)
    return face_encodings

# Function to create a database folder and store face encodings
def create_database(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
                input_image_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_image_path, input_directory)
                output_subfolder = os.path.join(output_directory, os.path.dirname(relative_path))
                output_image_path = os.path.join(output_subfolder, file)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)
                face_encodings = extract_face_encodings(input_image_path)
                if face_encodings:
                    np.save(output_image_path.replace('.jpg', ''), face_encodings[0])

# Example usage
input_directory = 'database1'
output_directory = 'database2'
create_database(input_directory, output_directory)
