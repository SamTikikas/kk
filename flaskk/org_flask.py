import numpy as np
import cv2

import scipy.io
from flask_cors import CORS
import tensorflow as tf
import os
from flask import Flask, request, jsonify
from keras.models import load_model
import base64
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import supervision as sv
import torchvision
import os

import timm

app = Flask(__name__)

CORS(app)

#Uploading DETR MODEL
MODEL_PATH = "detr-model"
model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
model.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
model.eval()

#Uploading UNET MODEL
model_path = "Unet.h5"
unet_model = load_model(model_path)

#UPLOADING UNETR MODEL
unet_r="unetr_100_16.h5"
unet_r_model=load_model(unet_r)



image_processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')

CONFIDENCE_THRESHOLD = 0.4
ANNOTATION_FILE_NAME = "_annotations.coco.json"


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        image_directory_path: str,
        image_processor,
        train: bool = True
    ):
        annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target



def segment_image(base64_image):
    try:
        # Decode base64 string to image
        img_data = base64.b64decode(base64_image)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        box_annotator = sv.BoxAnnotator()
        
        # utils.py
        TEST_DATASET = CocoDetection(image_directory_path="E:/flaskk/test", image_processor=image_processor, train=False)
        categories = TEST_DATASET.coco.cats
        id2label = {k: v['name'] for k, v in categories.items()}
        

    
        # Perform segmentation
        with torch.no_grad():
            inputs = image_processor(images=img, return_tensors='pt')
            outputs = model(**inputs)

            # post-process
            target_sizes = torch.tensor([img.shape[:2]])
            results = image_processor.post_process_object_detection(
                outputs=outputs,
                threshold=CONFIDENCE_THRESHOLD,
                target_sizes=target_sizes
            )[0]
          
            detections = sv.Detections.from_transformers(transformers_results=results)
            
            
            labels = [f"{id2label[class_id]} {confidence:.2f}" for _, _, confidence, class_id, _ in detections]
            
            
            frame_detections = box_annotator.annotate(scene=img.copy(), detections=detections, labels=labels)
          
            cv2.imwrite("E:/flaskk/results/dillo.jpg", frame_detections)
            _, buffer = cv2.imencode('.jpg', frame_detections) # Change to results if needed
            segmented_image_base64 = base64.b64encode(buffer).decode('utf-8')
            

        return segmented_image_base64

    except Exception as e:
        return str(e)


def remove_image(file_path):
    try:
        # Attempt to remove the file
        os.remove(file_path)
        print(f"Image at {file_path} successfully removed.")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
        
        
#UNET AND UNETR

global IMG_H
global IMG_W
global NUM_CLASSES
global CLASSES
global COLORMAP
IMG_H = 320
IMG_W = 416

RIMG_H=256
RIMG_W=256
NUM_CLASSES = 11


def get_colormap():
    
    colormap = scipy.io.loadmat("ultimate.mat")
    colormap = colormap["color"]
    classes = [
        "Background",
        "Spleen",
        "Right kidney",
        "Left kidney",
        "Liver",
        "Gallbladder",
        "Stomach",
        "Aorta",
        "Inferior vena cava",
        "Portal vein",
        "Pancreas"
    ]

    return classes, colormap


def grayscale_to_rgb(mask, classes, colormap):
    h, w, _ = mask.shape
    mask = mask.astype(np.int32)
    output = []

    for i, pixel in enumerate(mask.flatten()):
        output.append(colormap[pixel])

    output = np.reshape(output, (h, w, 3))
    return output



def read_image(x):
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMG_W, IMG_H))
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return x



    
def save_results(image, mask, pred, save_image_path,CLASSES,COLORMAP):
    image = image[0]
    h, w, _ = image.shape
    line = np.ones((10, w, 3)) * 255

    pred = np.expand_dims(pred, axis=-1)
    
    pred_rgb = grayscale_to_rgb(pred, CLASSES, COLORMAP)

    mask=mask[0]
    print(image.shape,"  ",pred_rgb.shape,"  ",mask.shape)

    alpha = 0.5
    blended_image1 = alpha * image + (1 - alpha) * mask
    blended_image2 = alpha * image + (1 - alpha) * pred_rgb


    # Concatenate images for visualization
    cat_images = np.concatenate([image, line, blended_image1, line,blended_image2],axis=0)
    cv2.imwrite(save_image_path, cat_images)
    return cat_images
    


def base64encode(img):
    _, buffer = cv2.imencode('.png', img)  # Change to results if needed
    segmented_image_base64 = base64.b64encode(buffer).decode('utf-8')
    return segmented_image_base64



def read_imageR(x):
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (RIMG_W, RIMG_H))
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return x



#Unet
@app.route('/segment', methods=['POST'])
def segment():
    
    
    CLASSES,COLORMAP=get_colormap()
    file = request.files['image']
    text_input = request.form.get('text_input', '')
    image_path = f'E:/flaskk/inputs/{text_input}.jpg'
    mask_path=f'E:/flaskk/mask_folder/{text_input}.png'
    file.save(image_path)
    

    #image_path = 'img0020.png'
    preprocessed_image = read_image(image_path)
    preprocessed_mask = read_image(mask_path)

    # Make prediction
    prediction = unet_model.predict(preprocessed_image,verbose=0)[0]
    prediction = np.argmax(prediction, axis=-1)
    prediction = prediction.astype(np.float32)


    save_image_path = "E:/flaskk/results/Unet.png" 
    tempo = save_results(preprocessed_image, preprocessed_mask, prediction, save_image_path,CLASSES,COLORMAP)


   
    segmented_image_base64=base64encode(tempo) 
    return jsonify({'segmented_image': segmented_image_base64})


#UNETR
@app.route('/segment_unetr', methods=['POST'])
def segmentr():
    
    
    
    CLASSES,COLORMAP=get_colormap()
    
    file = request.files['image']
    text_input = request.form.get('text_input', '')
    image_path = f'E:/flaskk/inputs/{text_input}.jpg'
    mask_path=f'E:/flaskk/mask_folder/{text_input}.png'
    file.save(image_path)
    

    #image_path = 'img0020.png'
    preprocessed_image = read_imageR(image_path)
    
    preprocessed_mask = read_imageR(mask_path)

    # Make prediction
    prediction = unet_r_model.predict(preprocessed_image,verbose=0)[0]
    prediction = np.argmax(prediction, axis=-1)
    prediction = prediction.astype(np.float32)

    # Post-process the mask
    # mask_ = read_mask(mask_path, COLORMAP)


    save_image_path = "E:/flaskk/results/untR.png" 
    tempo = save_results(preprocessed_image, preprocessed_mask, prediction, save_image_path,CLASSES,COLORMAP)


   
    segmented_image_base64=base64encode(tempo) 
    return jsonify({'segmented_image': segmented_image_base64})


#DETR
@app.route('/segment_detr', methods=['POST'])
def segmentdetr():
    try:
        base64_image = request.json['image']

        if base64_image:
            segmented_image_base64 = segment_image(base64_image)
            
            return jsonify({'segmented_image': segmented_image_base64})
        else:
            return jsonify({'error': 'Invalid image data'})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    results_folder = 'results'
    if not os.path.exists(results_folder):
            os.makedirs(results_folder)
    app.run(host='0.0.0.0', port=5000)
