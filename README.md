# Emotion Detection from Video Feed

This project uses a webcam to detect emotions in real-time using a pre-trained deep learning model. The application is built with Flask for the web interface and OpenCV for video processing.

## Dependencies

To run this project, you need the following Python packages:

- Flask
- OpenCV
- TensorFlow
- h5py
- numpy

You can install these dependencies using the provided `requirements.txt` file.

### Requirements File

Create a `requirements.txt` file with the following content:

```
Flask==2.0.3
opencv-python==4.5.3.20210927
tensorflow==2.6.0
h5py==3.1.0
numpy==1.21.2
```

## Setup Instructions

1. **Set Up a Virtual Environment (Optional but Recommended)**:

   - Create a virtual environment to keep your project dependencies isolated.

   ```bash
   python -m venv venv
   ```

   - Activate the virtual environment:
     - On Windows:
     ```bash
     venv\Scripts\activate
     ```
     - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

2. **Install Dependencies**:

   - Use pip to install the required packages from the `requirements.txt` file.

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:

   - Ensure you have the Haar Cascade file (`haarcascade_frontalface_default.xml`) in the same directory as your script or provide the correct path in your code.
   - Start the Flask application by running the `main.py` file:

   ```bash
   python main.py
   ```

4. **Access the Application**:
   - Open a web browser and go to `http://localhost:5000` to access the application.

## How It Works

- The application captures video from the webcam and processes each frame to detect faces.
- For each detected face, it predicts the emotion using a pre-trained model and displays the emotion label on the video feed.

## Notes

- Ensure that your camera is connected and accessible by OpenCV.
- If you encounter any issues with TensorFlow, make sure your system meets the requirements for the version you are installing, especially if you are using a GPU.
- You may need to adjust the versions of the packages in `requirements.txt` based on your specific environment or compatibility needs.
