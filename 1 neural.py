import cv2
# Сохраним оригинальные функции OpenCV до импорта ultralytics
orig_imshow = cv2.imshow
orig_namedWindow = cv2.namedWindow

from ultralytics import YOLO
import torch

print("CUDA available:", torch.cuda.is_available())

# Восстанавливаем оригинальные функции отображения
cv2.imshow = orig_imshow
cv2.namedWindow = orig_namedWindow

def main():
    # Загрузка модели YOLO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO("gun.pt")
    model.to(device)
    device = next(model.parameters()).device
    print("Модель запущена на устройстве:", device)

    # Захват видео с веб-камеры (0 – первая подключенная камера)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть веб-камеру.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось получить кадр.")
            break

        # Получение результатов обнаружения для текущего кадра
        results = model(frame)
        res = results[0]

        # Создаём копию кадра для аннотаций
        annotated_frame = frame.copy()

        # Если обнаружены объекты, обрабатываем боксы
        if res.boxes is not None and len(res.boxes) > 0:
            try:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy()
            except AttributeError:
                boxes = res.boxes.xyxy
                confs = res.boxes.conf
                classes = res.boxes.cls

            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                # Рисуем прямоугольник (цвет – зелёный)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{int(cls)} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Создаем окно для отображения (используем оригинальную функцию)
        cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("YOLO Detection", annotated_frame)

        # Выход из цикла по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
