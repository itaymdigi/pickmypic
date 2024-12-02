from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import cv2
from PIL import Image, ImageEnhance
import io
import numpy as np
import traceback
import logging
import pickle
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize face detector and recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Store reference faces with their images
reference_faces = {
    # name: {'encoding': encoding, 'image': image_data}
}

def encode_image(image):
    # Convert image to base64 string
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def enhance_image(pil_image):
    """Apply various enhancements to improve face detection"""
    try:
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.2)
        
        # Enhance brightness
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        return pil_image
    except Exception as e:
        logger.error(f"Error in enhance_image: {str(e)}")
        return pil_image

def detect_faces(image):
    """Detect faces using OpenCV"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Try different parameters for face detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            # Try with more lenient parameters
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=3,
                minSize=(20, 20)
            )
        
        return gray, faces
    except Exception as e:
        logger.error(f"Error in detect_faces: {str(e)}")
        return None, []

def extract_face_encoding(gray, face):
    """Extract face encoding using OpenCV's LBPH"""
    try:
        x, y, w, h = face
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (100, 100))  # Normalize size
        return face_roi
    except Exception as e:
        logger.error(f"Error in extract_face_encoding: {str(e)}")
        return None

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/api/upload-reference', methods=['POST', 'OPTIONS'])
def upload_reference():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        logger.info("=== Starting new upload request ===")
        
        if 'file' not in request.files:
            logger.warning("No file found in request")
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        name = request.form.get('name')
        
        logger.info(f"Processing file: {file.filename}")
        logger.info(f"Name provided: {name}")
        
        if not name:
            return jsonify({"error": "No name provided"}), 400
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        try:
            # Read and process image
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({'error': 'Invalid image file'}), 400

            # Convert to RGB for face detection
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(rgb_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                return jsonify({'error': 'No face detected in the image'}), 400
            
            if len(faces) > 1:
                return jsonify({'error': 'Multiple faces detected. Please use an image with a single face'}), 400

            # Get the face ROI
            x, y, w, h = faces[0]
            face_roi = rgb_image[y:y+h, x:x+w]
            
            # Create face encoding
            face_encoding = face_cascade.detectMultiScale(cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY))
            
            # Store both encoding and image
            reference_faces[name] = {
                'encoding': face_encoding,
                'image': encode_image(face_roi)  # Store the face ROI
            }
            
            return jsonify({
                'message': 'Reference face added successfully',
                'name': name,
                'faces_found': 1,
                'face_image': encode_image(face_roi)
            }), 200
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                "error": "Error processing image",
                "suggestion": "Please try a different photo"
            }), 500
            
    except Exception as e:
        logger.error(f"Error in upload_reference: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/recognize', methods=['POST', 'OPTIONS'])
def recognize_faces():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        logger.info("Received recognize request")
        
        if 'file' not in request.files:
            return jsonify({
                "error": "No file provided in the request",
                "files_received": list(request.files.keys())
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        try:
            # Read and process the image
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({'error': 'Invalid image file'}), 400

            # Convert to RGB for face detection
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(rgb_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                logger.warning("No faces detected")
                return jsonify({
                    "error": "No face detected in the image",
                    "suggestion": "Please ensure the image contains a clear, front-facing photo of a face"
                }), 400
            
            recognized_faces = []
            
            # For each face in the target image
            for (x, y, w, h) in faces:
                face_roi = rgb_image[y:y+h, x:x+w]
                face_encoding = face_cascade.detectMultiScale(cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY))
                
                # Compare with reference faces
                for name, ref_data in reference_faces.items():
                    ref_encoding = ref_data['encoding']
                    # Basic comparison - you might want to implement a more sophisticated matching algorithm
                    if len(face_encoding) > 0 and len(ref_encoding) > 0:
                        recognized_faces.append({
                            'name': name,
                            'reference_image': ref_data['image'],
                            'detected_face': encode_image(face_roi)
                        })
            
            return jsonify({
                'faces_found': len(faces),
                'recognized_faces': recognized_faces
            }), 200
        
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Error in recognize_faces: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
