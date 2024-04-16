from flask import Flask, render_template, request
import cv2
import json
import base64
import pytesseract
import os
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from scipy.ndimage import rotate as Rotate 


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

app = Flask(__name__)

crop_dir_name = "ultralytics_crop"
if not os.path.exists(crop_dir_name):
    os.mkdir(crop_dir_name)

model = YOLO(r'C:\Users\hrishikesh.atole\Desktop\Anaconda testing\flask 2\best.pt')
# Encode the image to base64 for displaying in HTML
def get_resized_base64_image(image, width=None, height=None):
        if width is not None and height is not None:
            html_image = cv2.resize(image, (width, height))
        elif width is not None:
            aspect_ratio = float(image.shape[1]) / float(image.shape[0])
            html_image = cv2.resize(image, (width, int(width / aspect_ratio)))
        elif height is not None:
            aspect_ratio = float(image.shape[1]) / float(image.shape[0])
            html_image = cv2.resize(image, (int(height * aspect_ratio), height))
        else:
            html_image = image
        _, encoded_image = cv2.imencode('.jpg', html_image)
        encoded_image_str = base64.b64encode(encoded_image).decode('utf-8')
        return encoded_image_str

def class_0(cropped_image):
    custom_config = r'--psm 6 --oem 3 -l eng' 
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    adress = pytesseract.image_to_string(cropped_image, config=custom_config)
    return(adress)


def class_1(cropped_image):
    custom_config = r'--psm 6 --oem 3 -l eng' 
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    class_txt = pytesseract.image_to_string(cropped_image, config=custom_config)
    return(class_txt)
    
def class_2(cropped_image):
    
    custom_config = r'--psm 6 --oem 3 -l eng' 
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    dob = pytesseract.image_to_string(cropped_image, config=custom_config)
    return(dob)

def class_3(cropped_image):
    custom_config = r'--psm 6 --oem 3 -l eng' 
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    exp_date = pytesseract.image_to_string(cropped_image,custom_config)
    return(exp_date)

def class_4(cropped_image):
    custom_config = r'--psm 6 --oem 3 -l eng' 
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    first_name = pytesseract.image_to_string(cropped_image, custom_config)
    return (first_name)

def class_5(cropped_image):
    custom_config = r'--psm 6 --oem 3 -l eng' 
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    issu_date = pytesseract.image_to_string(cropped_image, config=custom_config)
    return(issu_date)

def class_6(cropped_image):
    
    custom_config = r'--psm 6 --oem 3 -l eng' 
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    last_name = pytesseract.image_to_string(cropped_image, config=custom_config)
    return(last_name)

def class_7(cropped_image):
    custom_config = r'--psm 6 --oem 3 -l eng' 
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    linc = pytesseract.image_to_string(cropped_image, config=custom_config)
    return(linc)

def class_8(cropped_image):
    custom_config = r'--psm 6 --oem 3 -l eng' 
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    sex = pytesseract.image_to_string(cropped_image, config=custom_config)
    return(sex)

#Imafe rotation
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def float_converter(x):
    if x.isdigit():
        out = float(x)
    else:
        out = x
    return out 

#def find_rotation(img: str):
def find_rotation(img):
    #img = cv2.imread(img) if isinstance(img, str) else img
    orientation_info = pytesseract.image_to_osd(img)
    orientation_data = {i.split(":")[0].strip(): float_converter(i.split(":")[-1].strip()) for i in orientation_info.rstrip().split("\n")}
    rotation_angle = 360 - orientation_data["Rotate"]
    img_rotated = Rotate(img, rotation_angle)
    return img_rotated

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    
    uploaded_file = request.files['file']

    if not uploaded_file:
        return "No file uploaded", 400

    # Read the uploaded image
    try:
        image_buffer = uploaded_file.read()
        nparr = np.frombuffer(image_buffer, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return f"Error reading image: {str(e)}", 500

    if original_image is None:
        return "Error decoding image", 500
    

    
    img_rotate = find_rotation(original_image)
    results = model.predict(img_rotate, save=False, imgsz=320, conf=0.5)
    names = model.names
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    
    #annotator = Annotator(original_image, line_width=2, example=names)
    
    outputs = [] 
    
    for box, cls in zip(boxes, clss):
        #annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])
        
        crop_obj = img_rotate[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        
        if cls == 0:
            output = class_0(crop_obj)
            cleaned_output = output.replace('\n', '').replace('\x0c', '')
            info = {
                'Address':cleaned_output,
            }
            outputs.append(info)
        elif cls == 1:
            output = class_1(crop_obj)
            cleaned_output = output.replace('\n', '').replace('\x0c', '')
            info = {
            'Class':cleaned_output,
            }
            outputs.append(info)
        elif cls == 2:
            output = class_2(crop_obj)
            cleaned_output = output.replace('\n', '').replace('\x0c', '')
            info = {
                'DOB':cleaned_output,
            }
            outputs.append(info)
        elif cls == 3:
            output = class_3(crop_obj)
            cleaned_output = output.replace('\n', '').replace('\x0c', '')
            info = {
                'Exp_Date':cleaned_output,
            }
            outputs.append(info)
        elif cls == 4:
            output = class_4(crop_obj)
            cleaned_output = output.replace('\n', '').replace('\x0c', '')
            info = {
               'First Name':cleaned_output,
            }
            outputs.append(info)
        elif cls == 5:
            output = class_5(crop_obj)
            cleaned_output = output.replace('\n', '').replace('\x0c', '')
            info = {
                'Issue Date':cleaned_output,
            }
            outputs.append(info)
        elif cls == 6:
            output = class_6(crop_obj)
            cleaned_output = output.replace('\n', '').replace('\x0c', '')
            info = {
                'Last Name':cleaned_output,
            }
            outputs.append(info)
        elif cls == 7:
            output = class_7(crop_obj)
            cleaned_output = output.replace('\n', '').replace('\x0c', '')
            info = {
                'Lincens Number':cleaned_output,
            }
            outputs.append(info)
        elif cls == 8:
            output = class_8(crop_obj)
            cleaned_output = output.replace('\n', '').replace('\x0c', '')
            info = {
                'Sex':cleaned_output,
            }
            outputs.append(info)
        

    outputs_json = json.dumps(outputs)
    json_filename = r'flask3\output.json'
    with open(json_filename, 'w') as json_file:
        json_file.write(outputs_json)

    encoded_image_str = get_resized_base64_image(original_image, width=400) 

    return render_template('result.html', encoded_image=encoded_image_str, outputs_json=outputs_json, json_filename=json_filename)

if __name__ == '__main__':
    app.run(debug=True)

