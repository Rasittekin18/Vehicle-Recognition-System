import cv2
import asyncio
import json
import os
from fastanpr import FastANPR

# FastANPR sınıfından bir örnek oluştur
fast_anpr = FastANPR()


async def main():
    # Resim dosyalarının yollarını belirle
    image_files = [
        "C:\\Users\\Rasit\\PycharmProjects\\pythonProject9\\vehicle_detection_tracker-main\\examples\\10.jpg",
        # Klasördeki diğer resim dosyalarını da buraya ekleyebilirsiniz
    ]

    # Resimleri yükle
    images = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in image_files]

    # ANPR işlemini çalıştır
    number_plates = await fast_anpr.run(images)

    # JSON çıktısı için veri yapısını oluştur
    output_data = {}

    # Sonuçları işleme
    for file, plates in zip(image_files, number_plates):
        # JSON dosyasına eklemek için her resim için veri hazırlama
        file_data = []
        # Orijinal resmi yeniden yükle (renkli hali)
        image = cv2.imread(file)

        # Görüntüyü 500x500 olarak yeniden boyutlandır
        target_size = (500, 500)
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

        # Oranları hesapla
        x_ratio = target_size[0] / image.shape[1]
        y_ratio = target_size[1] / image.shape[0]

        # Plakaların kutu bilgilerini incele
        for plate in plates:
            plate_info = {
                "detection_bounding_box": plate.det_box,
                "detection_confidence": plate.det_conf,
                "recognition_text": plate.rec_text,
                "recognition_polygon": plate.rec_poly,
                "recognition_confidence": plate.rec_conf
            }
            file_data.append(plate_info)

            # Plaka kutusunu çiz
            box = plate.det_box
            if isinstance(box, list) and len(box) == 4:
                top_left = (int(box[0] * x_ratio), int(box[1] * y_ratio))
                bottom_right = (int(box[2] * x_ratio), int(box[3] * y_ratio))
                cv2.rectangle(resized_image, top_left, bottom_right, (0, 255, 0), 2)

                # Tanımlı metni resim üzerine yaz
                text = plate.rec_text
                cv2.putText(resized_image, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)

        # JSON çıktısına ekle
        output_data[file] = file_data

        # Yeniden boyutlandırılmış görseli göster
        cv2.imshow('ANPR Results', resized_image)
        cv2.waitKey(0)  # Her görselin gösterilmesi için bir tuşa basmayı bekler
        cv2.destroyAllWindows()

    # JSON dosyasının mevcut olup olmadığını kontrol et
    json_file_path = 'anpr_results.json'
    if not os.path.exists(json_file_path):
        # JSON dosyası yoksa oluştur ve veriyi yaz
        with open(json_file_path, 'w') as json_file:
            json.dump(output_data, json_file, indent=4)
    else:
        # JSON dosyası varsa, mevcut veriyi yükle ve yeni veriyi ekle
        with open(json_file_path, 'r') as json_file:
            existing_data = json.load(json_file)

        # Yeni veriyi mevcut veriye ekle
        existing_data.update(output_data)

        # Güncellenmiş veriyi JSON dosyasına yaz
        with open(json_file_path, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)


# Asenkron işlevi çalıştırın
if __name__ == "__main__":
    asyncio.run(main())
