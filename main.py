import cv2
import numpy as np


def load_yolo_model(config_path, weights_path):
    """
    Загружает сеть YOLO по конфигурационному файлу и файлу весов.
    """
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    return net


def get_output_layers(net):
    """
    Получает имена выходных слоев сети YOLO.
    """
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes):
    """
    Рисует прямоугольник и подписывает обнаруженный объект.
    """
    label = str(classes[class_id]) if classes is not None and class_id < len(classes) else str(class_id)
    color = (0, 255, 0)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main():
    # Пути к файлам: конфигурация, веса и классы
    config_path = "yolov11.cfg"
    weights_path = "yolov11.weights"
    classes_file = "coco.names"  # или другой файл с именами классов

    # Загрузка списка классов
    try:
        with open(classes_file, "r") as f:
            classes = [line.strip() for line in f.readlines()]
    except Exception as e:
        print("Не удалось загрузить файл с классами:", e)
        classes = None

    # Загрузка модели YOLO
    net = load_yolo_model(config_path, weights_path)

    # Загрузка тестового изображения
    image_path = "test.jpg"  # замените на путь к вашему изображению
    image = cv2.imread(image_path)
    if image is None:
        print("Не удалось загрузить изображение:", image_path)
        return

    Height, Width = image.shape[:2]

    # Преобразование изображения в blob
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Получаем имена выходных слоев и выполняем прямой проход
    output_layers = get_output_layers(net)
    outs = net.forward(output_layers)

    conf_threshold = 0.5  # порог уверенности
    nms_threshold = 0.4  # порог для non-max suppression

    class_ids = []
    confidences = []
    boxes = []

    # Обработка выходных данных сети
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Применяем non-max suppression для устранения перекрывающихся обнаружений
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        x, y, w, h = boxes[i]
        draw_prediction(image, class_ids[i], confidences[i], x, y, x + w, y + h, classes)

    # Отображение результата
    cv2.imshow("Результат", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
