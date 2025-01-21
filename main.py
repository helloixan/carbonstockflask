from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import os
from PIL import Image
import io

app = Flask(__name__)

yolo_model = YOLO('carbondetectv3.pt')

midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu').eval()

transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

def calc_carbon(height):
    a = 0.05
    b = 2.45
    ptg_organic = 0.47
    Cv = (a * (height)**b) * ptg_organic
    return Cv

@app.route("/predict/", methods=["POST"])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No file part"}), 400

        image = request.files['image']
        image_data = image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        H, W, _ = img.shape

        results = yolo_model(img)
        bounding_boxes = []
        for box in results[0].boxes.xyxy:
            x_min, y_min, x_max, y_max = map(int, box.tolist())
            bounding_boxes.append((x_min, y_min, x_max, y_max))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img_rgb).to('cpu')
        with torch.no_grad():
            prediction = midas(img_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze().cpu().numpy()

        scaling_factor = 0.01
        results_data = []
        for x_min, y_min, x_max, y_max in bounding_boxes:
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            depth_at_center = prediction[center_y, center_x]
            depth_in_meters = depth_at_center * scaling_factor
            height_px = y_max - y_min
            pixel_to_meter_ratio = depth_in_meters / H
            height_actual = height_px * pixel_to_meter_ratio * 10
            carbon = calc_carbon(height_actual)

            results_data.append({
                "center": (center_x, center_y),
                "depth_in_meters": float(depth_in_meters),
                "height_actual": float(height_actual),
                "carbon": float(carbon)
            })

        return jsonify(results_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
