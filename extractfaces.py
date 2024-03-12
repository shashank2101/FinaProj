import os
import cv2
from mtcnn import MTCNN

# Function to detect faces and save cropped images
def detect_and_save_faces(input_dir, output_dir,target_size=(224,224)):
    # Initialize MTCNN detector
    detector = MTCNN()

    # Traverse through folders in the input directory
    for person_folder in os.listdir(input_dir):
        person_folder_path = os.path.join(input_dir, person_folder)
        if os.path.isdir(person_folder_path):
            # Create corresponding person folder in the output directory
            output_person_folder = os.path.join(output_dir, person_folder)
            os.makedirs(output_person_folder, exist_ok=True)

            # Traverse through images in the person's folder
            for image_file in os.listdir(person_folder_path):
                image_path = os.path.join(person_folder_path, image_file)
                if os.path.isfile(image_path):
                    # Read the image
                    image = cv2.imread(image_path)
                    if image is not None:
                        # Detect faces in the image
                        result = detector.detect_faces(image)
                        if result:
                            for i, face_data in enumerate(result):
                                # Get bounding box coordinates
                                x, y, w, h = face_data['box']

                                # Crop face region from the image
                                face = image[y:y+h, x:x+w]

                                # Save cropped face image
                                output_file_path = os.path.join(output_person_folder, f"{image_file}")
                                resized_face = cv2.resize(face, target_size)
                                cv2.imwrite(output_file_path, resized_face)

# Define input and output directories
input_database_dir = 'database'
output_database_dir = 'database1'

# Call the function to detect faces and save cropped images
detect_and_save_faces(input_database_dir, output_database_dir)
