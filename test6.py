import cv2
import numpy as np
import os
import json
from datetime import datetime
from PIL import Image
from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker
from fastanpr import FastANPR
import asyncio

# Define the vehicle types to display
vehicle_types_to_display = {"car", "bus", "truck", "minibus", "lorry", "motorcycle", "ship", "taxi"}

# Read JSON file
def read_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        return json.load(json_file)

# Detect vehicles
def detect_vehicles(vehicle_detection, img_array):
    response = vehicle_detection.process_frame(img_array, datetime.now())
    if 'detected_vehicles' in response and response['detected_vehicles']:
        return response
    return None

# Recognize plates
async def recognize_plates(fast_anpr, image_arrays):
    number_plates = await fast_anpr.run(image_arrays)
    return number_plates

# Update JSON file
def update_json(json_file_path, new_data):
    with open(json_file_path, 'w') as json_file:
        json.dump(new_data, json_file, indent=4)

# Main function
async def main():
    # JSON file paths
    json_file_path = 'veri2.json'
    output_json_path = 'results2.json'

    # Read JSON data
    data = read_json(json_file_path)

    # Initialize results list
    results = []

    for entry in data:
        file_path = entry["filePath"]
        doc_name = entry["DocName"]

        # Initialize classes
        vehicle_detection = VehicleDetectionTracker()
        fast_anpr = FastANPR()

        # Open and process image
        img = Image.open(file_path)
        img = img.convert('RGB')
        img_array = np.array(img)

        # Detect vehicles
        detection_result = detect_vehicles(vehicle_detection, img_array)

        vehicles_info = []
        plates_info = []

        result_data = {
            "filePath": file_path,
            "docName": doc_name,
            "vehicles": [],
            "plates": []
        }

        if detection_result:
            cropped_images = []

            for idx, vehicle in enumerate(detection_result['detected_vehicles']):
                vehicle_type = vehicle.get("vehicle_type", "").lower()
                if vehicle_type not in vehicle_types_to_display:
                    continue

                bbox = vehicle["vehicle_coordinates"]
                x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["width"]), int(bbox["height"])

                cropped_vehicle = img_array[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
                if cropped_vehicle.size > 0:
                    cropped_images.append(cropped_vehicle)

                    color_info = vehicle.get("color_info", "Unknown")
                    model_info = vehicle.get("model_info", "Unknown")

                    vehicles_info.append({
                        "vehicle_id": vehicle["vehicle_id"],
                        "vehicle_type": vehicle["vehicle_type"],
                        "detection_confidence": vehicle["detection_confidence"],
                        "color_info": vehicle["color_info"],
                        "model_info": vehicle["model_info"],
                        "bbox": {"x": x, "y": y, "width": w, "height": h}
                    })

            number_plates = await recognize_plates(fast_anpr, cropped_images)
            plates_info_dict = {}

            for idx, plates in enumerate(number_plates):
                vehicle_id = detection_result['detected_vehicles'][idx]["vehicle_id"]
                for plate in plates:
                    if vehicle_id not in plates_info_dict or (
                            plate.det_conf is not None and plate.det_conf > plates_info_dict[vehicle_id][
                        "detection_confidence"]):
                        bbox = plate.det_box if plate.det_box is not None else [0, 0, 0, 0]
                        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

                        plates_info_dict[vehicle_id] = {
                            "detection_confidence": round(plate.det_conf, 2) if plate.det_conf is not None else None,
                            "recognition_text": plate.rec_text,
                            "recognition_confidence": round(plate.rec_conf, 2) if plate.rec_conf is not None else None,
                            "detection_bbox": {
                                "xmin": int(xmin),
                                "ymin": int(ymin),
                                "xmax": int(xmax),
                                "ymax": int(ymax)
                            }
                        }

            combined_result = []
            for vehicle in vehicles_info:
                vehicle_id = vehicle["vehicle_id"]
                plate_info = plates_info_dict.get(vehicle_id, {})
                vehicle.update(plate_info)
                combined_result.append(vehicle)

            # Check if any vehicle has a detected plate
            if not any(vehicle.get("recognition_text") for vehicle in combined_result):
                # If no plates detected, process the original image
                original_number_plates = await recognize_plates(fast_anpr, [img_array])
                for plate in original_number_plates[0]:
                    if plate.det_conf is not None and plate.rec_text:
                        bbox = plate.det_box if plate.det_box is not None else [0, 0, 0, 0]
                        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
                        plates_info.append({
                            "vehicle_id": None,
                            "vehicle_type": None,
                            "detection_confidence": round(plate.det_conf, 2) if plate.det_conf is not None else None,
                            "color_info": None,
                            "model_info": None,
                            "bbox": None,
                            "recognition_text": plate.rec_text,
                            "recognition_confidence": round(plate.rec_conf, 2) if plate.rec_conf is not None else None,
                            "detection_bbox": {
                                "xmin": int(xmin),
                                "ymin": int(ymin),
                                "xmax": int(xmax),
                                "ymax": int(ymax)
                            }
                        })

            result_data["vehicles"] = combined_result
            result_data["plates"] = plates_info

        else:
            # No vehicles detected, process the original image
            number_plates = await recognize_plates(fast_anpr, [img_array])
            plates_result = []

            for plates in number_plates:
                for plate in plates:
                    bbox = plate.det_box if plate.det_box is not None else [0, 0, 0, 0]
                    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
                    plates_result.append({
                        "vehicle_id": None,
                        "vehicle_type": None,
                        "detection_confidence": round(plate.det_conf, 2) if plate.det_conf is not None else None,
                        "color_info": None,
                        "model_info": None,
                        "bbox": None,
                        "recognition_text": plate.rec_text,
                        "recognition_confidence": round(plate.rec_conf, 2) if plate.rec_conf is not None else None,
                        "detection_bbox": {
                            "xmin": int(xmin),
                            "ymin": int(ymin),
                            "xmax": int(xmax),
                            "ymax": int(ymax)
                        }
                    })

            result_data["vehicles"] = []
            result_data["plates"] = plates_result

        # Add results to the list
        results.append(result_data)

    # Save results to JSON file
    update_json(output_json_path, results)
    print(f"Results saved to JSON file: {output_json_path}")

# Run asynchronous function
if __name__ == "__main__":
    asyncio.run(main())
