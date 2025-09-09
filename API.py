from fastapi import FastAPI, UploadFile, File
import json
import numpy as np
from PIL import Image
from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker
from fastanpr import FastANPR
import asyncio
from io import BytesIO
import os
from datetime import datetime

# FastAPI uygulamasını oluşturuyoruz
app = FastAPI()

# Araç türleri
vehicle_types_to_display = {"car", "bus", "truck", "minibus", "lorry", "motorcycle", "ship", "taxi"}

# Araç tespit fonksiyonu
def detect_vehicles(vehicle_detection, img_array):
    response = vehicle_detection.process_frame(img_array, datetime.now())
    if 'detected_vehicles' in response and response['detected_vehicles']:
        return response
    return None

# Plaka tanıma fonksiyonu
async def recognize_plates(fast_anpr, image_arrays):
    number_plates = await fast_anpr.run(image_arrays)
    return number_plates

# Resim işleme endpoint'i (Fotoğraf Yükleme)
@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert('RGB')
    img_array = np.array(img)

    vehicle_detection = VehicleDetectionTracker()
    fast_anpr = FastANPR()

    detection_result = detect_vehicles(vehicle_detection, img_array)
    result_data = {"vehicles": [], "plates": []}
    vehicles_info = []
    plates_info = []

    if detection_result:
        cropped_images = []
        for vehicle in detection_result['detected_vehicles']:
            vehicle_type = vehicle.get("vehicle_type", "").lower()
            if vehicle_type not in vehicle_types_to_display:
                continue
            bbox = vehicle["vehicle_coordinates"]
            x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["width"]), int(bbox["height"])
            cropped_vehicle = img_array[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
            if cropped_vehicle.size > 0:
                cropped_images.append(cropped_vehicle)
                vehicles_info.append({
                    "vehicle_id": vehicle["vehicle_id"],
                    "vehicle_type": vehicle["vehicle_type"],
                    "detection_confidence": vehicle["detection_confidence"],
                    "color_info": vehicle.get("color_info", "Unknown"),
                    "model_info": vehicle.get("model_info", "Unknown"),
                    "bbox": {"x": x, "y": y, "width": w, "height": h}
                })

        number_plates = await recognize_plates(fast_anpr, cropped_images)
        plates_info_dict = {}
        for idx, plates in enumerate(number_plates):
            vehicle_id = detection_result['detected_vehicles'][idx]["vehicle_id"]
            for plate in plates:
                bbox = plate.det_box if plate.det_box is not None else [0, 0, 0, 0]
                plates_info_dict[vehicle_id] = {
                    "recognition_text": plate.rec_text,
                    "recognition_confidence": round(plate.rec_conf, 2) if plate.rec_conf is not None else None,
                    "detection_bbox": {"xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3]}
                }

        for vehicle in vehicles_info:
            vehicle_id = vehicle["vehicle_id"]
            plate_info = plates_info_dict.get(vehicle_id, {})
            vehicle.update(plate_info)
        result_data["vehicles"] = vehicles_info

        if not any(vehicle.get("recognition_text") for vehicle in vehicles_info):
            original_number_plates = await recognize_plates(fast_anpr, [img_array])
            for plate in original_number_plates[0]:
                bbox = plate.det_box if plate.det_box is not None else [0, 0, 0, 0]
                plates_info.append({
                    "recognition_text": plate.rec_text,
                    "recognition_confidence": round(plate.rec_conf, 2) if plate.rec_conf is not None else None,
                    "detection_bbox": {"xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3]}
                })
            result_data["plates"] = plates_info

    else:
        number_plates = await recognize_plates(fast_anpr, [img_array])
        plates_result = []
        for plates in number_plates:
            for plate in plates:
                bbox = plate.det_box if plate.det_box is not None else [0, 0, 0, 0]
                plates_result.append({
                    "recognition_text": plate.rec_text,
                    "recognition_confidence": round(plate.rec_conf, 2) if plate.rec_conf is not None else None,
                    "detection_bbox": {"xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3]}
                })
        result_data["plates"] = plates_result

    return result_data


# JSON işleme endpoint'i (JSON içindeki filePath'lerle çalışma)
@app.post("/process_json/")
async def process_json(file: UploadFile = File(...)):
    contents = await file.read()
    data = json.loads(contents)

    results = []
    for entry in data:
        file_path = entry["filePath"]

        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}

        img = Image.open(file_path).convert('RGB')
        img_array = np.array(img)

        vehicle_detection = VehicleDetectionTracker()
        fast_anpr = FastANPR()

        detection_result = detect_vehicles(vehicle_detection, img_array)
        result_data = {"vehicles": [], "plates": []}
        vehicles_info = []
        plates_info = []

        if detection_result:
            cropped_images = []
            for vehicle in detection_result['detected_vehicles']:
                vehicle_type = vehicle.get("vehicle_type", "").lower()
                if vehicle_type not in vehicle_types_to_display:
                    continue
                bbox = vehicle["vehicle_coordinates"]
                x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["width"]), int(bbox["height"])
                cropped_vehicle = img_array[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
                if cropped_vehicle.size > 0:
                    cropped_images.append(cropped_vehicle)
                    vehicles_info.append({
                        "vehicle_id": vehicle["vehicle_id"],
                        "vehicle_type": vehicle["vehicle_type"],
                        "detection_confidence": vehicle["detection_confidence"],
                        "color_info": vehicle.get("color_info", "Unknown"),
                        "model_info": vehicle.get("model_info", "Unknown"),
                        "bbox": {"x": x, "y": y, "width": w, "height": h}
                    })

            number_plates = await recognize_plates(fast_anpr, cropped_images)
            plates_info_dict = {}
            for idx, plates in enumerate(number_plates):
                vehicle_id = detection_result['detected_vehicles'][idx]["vehicle_id"]
                for plate in plates:
                    bbox = plate.det_box if plate.det_box is not None else [0, 0, 0, 0]
                    plates_info_dict[vehicle_id] = {
                        "recognition_text": plate.rec_text,
                        "recognition_confidence": round(plate.rec_conf, 2) if plate.rec_conf is not None else None,
                        "detection_bbox": {"xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3]}
                    }

            for vehicle in vehicles_info:
                vehicle_id = vehicle["vehicle_id"]
                plate_info = plates_info_dict.get(vehicle_id, {})
                vehicle.update(plate_info)
            result_data["vehicles"] = vehicles_info

            if not any(vehicle.get("recognition_text") for vehicle in vehicles_info):
                original_number_plates = await recognize_plates(fast_anpr, [img_array])
                for plate in original_number_plates[0]:
                    bbox = plate.det_box if plate.det_box is not None else [0, 0, 0, 0]
                    plates_info.append({
                        "recognition_text": plate.rec_text,
                        "recognition_confidence": round(plate.rec_conf, 2) if plate.rec_conf is not None else None,
                        "detection_bbox": {"xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3]}
                    })
                result_data["plates"] = plates_info

        else:
            number_plates = await recognize_plates(fast_anpr, [img_array])
            plates_result = []
            for plates in number_plates:
                for plate in plates:
                    bbox = plate.det_box if plate.det_box is not None else [0, 0, 0, 0]
                    plates_result.append({
                        "recognition_text": plate.rec_text,
                        "recognition_confidence": round(plate.rec_conf, 2) if plate.rec_conf is not None else None,
                        "detection_bbox": {"xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3]}
                    })
            result_data["plates"] = plates_result

        results.append(result_data)

    return results


# Uvicorn sunucusunu başlatmak için
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
