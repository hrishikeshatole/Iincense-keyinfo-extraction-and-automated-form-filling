## Heads Up!

Hey there, thanks for visiting this repo! Before you dive into this project, just a quick heads up: it's more of a fun experiment than a polished, production-ready app. Think of it as a sneak peek into what's possible with OCR, Flask, and Yolo!

## What does that mean for you?  
Well, it means the accuracy might not be 100%.

## To answer it: 
The reason is I have train yolo's YOLOv8n.pt for training and only for 10 epochs. 

## Flask OCR Application

This is a Flask application for performing Optical Character Recognition (OCR) on uploaded images. The application utilizes the YOLO (You Only Look Once) object detection model to detect text regions in the image, and then performs OCR using Tesseract OCR.

## Features

- Upload an image containing text.
- Detect text regions using YOLO.
- Perform OCR on detected text regions.
- Extract information such as address, class, date of birth, expiration date, first name, issue date, last name, license number, and sex from the text.
- Display the original image with annotated text regions and extracted information.
- Save the extracted information in a JSON file.

## Installation

1. Clone the repository or download it.

2. Install the required dependencies

3. Download the YOLO pre-trained model weights (`.pt` file) and place it in the project directory.

## Usage

1. Start the Flask server: "python app.py"  
2. Open a web browser and navigate to `http://localhost:5000`.
3. Upload an image using the provided form.
4. View the annotated image with extracted information.

## File Structure

- `app.py`: The main Flask application file.
- `index.html`: HTML template for the upload page.
- `result.html`: HTML template for displaying the annotated image and extracted information.
- `templates/`: Directory for storing HTML templates.
- `ultralytics_crop/`: Directory for storing cropped text regions.

## Dependencies

- Flask
- OpenCV (cv2)
- Numpy
- PyTesseract
- Ultralytics YOLO
- Scipy
