# PickMyPic - Face Recognition Web Application

A web application that allows users to upload reference faces and perform face recognition on target images using OpenCV.

## Features

- Upload reference faces with names
- Detect faces in target images
- Match detected faces against reference faces
- Display results with reference and detected face images
- Modern, responsive UI with step-by-step workflow

## Tech Stack

### Backend
- Python 3.10+
- Flask
- OpenCV
- NumPy
- Pillow (PIL)

### Frontend
- React
- TypeScript
- Tailwind CSS
- React Hot Toast
- Lucide React Icons

## Setup

### Backend Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
venv\Scripts\activate
```
- Unix/MacOS:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Flask application:
```bash
cd backend
python app.py
```

The backend server will start at `http://localhost:5000`

### Frontend Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm start
```

The frontend application will start at `http://localhost:3000`

## API Endpoints

### POST /upload
Upload a reference face image with a name.

Request:
- Form data:
  - `file`: Image file
  - `name`: Name of the person

Response:
```json
{
  "message": "Face added successfully",
  "name": "John Doe",
  "faces_found": 1,
  "face_image": "base64_encoded_image"
}
```

### POST /recognize
Upload an image to recognize faces.

Request:
- Form data:
  - `file`: Image file

Response:
```json
{
  "faces_found": 2,
  "recognized_faces": [
    {
      "name": "John Doe",
      "reference_image": "base64_encoded_image",
      "detected_face": "base64_encoded_image"
    }
  ]
}
```

## Development

### Backend Structure
```
backend/
├── app.py          # Main Flask application
├── requirements.txt # Python dependencies
```

### Frontend Structure
```
frontend/
├── src/
│   ├── App.tsx    # Main React component
│   ├── App.css    # Styles
│   └── index.tsx  # Entry point
├── package.json
└── tailwind.config.js
```

## License

MIT License
