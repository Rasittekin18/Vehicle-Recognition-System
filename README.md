# Vehicle Recognition System
- This project is a Vehicle Recognition System capable of detecting and classifying vehicles by their license plate, brand, model, color, and type. The system processes images, extracts vehicle information, and outputs the results in JSON format, making it compatible with large-scale AI and LLM integrations.


🚀 Features

- License Plate Recognition (LPR):
- Uses FastANPR library for automatic plate detection.
- YOLOv8 model for locating license plates.
- PaddleOCR for optical character recognition.
- Results categorized as:
🟢 Correctly recognized
⚪ Partially recognized
🔴 Not recognized


- Vehicle Attribute Classification:
- Detects vehicle brand, model, color, and type.
- Vehicle location & type detection with YOLOv8.
- Brand & color classification with MobileNetV3-based models.

- JSON Output:
- All recognition results are stored in structured JSON format.
- Easy to integrate with other systems such as LLMs or databases.

- Evaluation Results:
- Tested on 80 images containing 130 vehicles.
- 71 vehicles had visible license plates.
- In another experiment, 78 images with 148 vehicles were tested.
- The system outputs both vehicle attributes and license plate recognition results.

🛠️ Technologies Used

- Python
- YOLOv8 (Ultralytics)
- FastANPR
- PaddleOCR
- MobileNetV3


Şunu dedin:
Vehicle Recognition System.pdf
PDF
Bu dosyayı incele ve github için bir readme dosyası yaz güzelce 
ChatGPT:

Senin yüklediğin PDF sunumunu inceledim ve GitHub için detaylı, profesyonel bir README.md dosyası hazırladım. Aşağıdaki metni direkt olarak README.md dosyana ekleyebilirsin.

Vehicle Recognition System

This project is a Vehicle Recognition System capable of detecting and classifying vehicles by their license plate, brand, model, color, and type. The system processes images, extracts vehicle information, and outputs the results in JSON format, making it compatible with large-scale AI and LLM integrations.

🚀 Features

License Plate Recognition (LPR):

Uses FastANPR library for automatic plate detection.

YOLOv8 model for locating license plates.

PaddleOCR for optical character recognition.

Results categorized as:

🟢 Correctly recognized

⚪ Partially recognized

🔴 Not recognized

Vehicle Attribute Classification:

Detects vehicle brand, model, color, and type.

Vehicle location & type detection with YOLOv8.

Brand & color classification with MobileNetV3-based models.

JSON Output:

All recognition results are stored in structured JSON format.

Easy to integrate with other systems such as LLMs or databases.

Evaluation Results:

Tested on 80 images containing 130 vehicles.

71 vehicles had visible license plates.

In another experiment, 78 images with 148 vehicles were tested.

The system outputs both vehicle attributes and license plate recognition results.

🛠️ Technologies Used

Python

YOLOv8 (Ultralytics)

FastANPR

PaddleOCR

MobileNetV3

📂 Project Structure

├── data/              # Sample images and datasets
├── models/            # YOLOv8 and MobileNetV3 models
├── outputs/           # JSON outputs and result images
├── src/               # Source code
│   ├── detection/     # Vehicle detection & tracking
│   ├── classification # Brand, model, color classifiers
│   └── ocr/           # License plate recognition
├── README.md          # Project documentation


📊 Example Output (JSON)

{
  "vehicle_id": 1,
  "brand": "Toyota",
  "model": "Corolla",
  "color": "White",
  "type": "Sedan",
  "license_plate": "34ABC123",
  "status": "recognized"
}

📷 Sample Results

- Detects and tracks vehicles in input images.
- Extracts license plates and runs OCR.
- Provides structured outputs for further processing.



✅ Conclusion
- The system demonstrates high accuracy in both vehicle classification and license plate recognition. It can be further integrated into traffic monitoring, smart city applications, and AI-driven analytics systems.


























