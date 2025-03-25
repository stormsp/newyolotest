from ultralytics import YOLO
import cv2

def draw_boxes(annotated_frame, res, color, class_names, filter_class=None):
    """
    Отрисовка боксов на кадре с возможностью фильтрации по классу.
    Вместо названия модели выводится имя класса, определённого нейросетью.
    """
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
            # Если фильтр задан, пропускаем боксы, не соответствующие нужному классу
            if filter_class is not None and int(cls) != filter_class:
                continue
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            # Получаем имя класса из словаря class_names, если нет - выводим id
            class_name = class_names.get(int(cls), str(int(cls)))
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    # Список имён файлов моделей (весов)
    #model_names = ["person.pt"]
    model_names = ["fire.pt", "person.pt", "gun.pt", "flood.pt", "tree.pt", "cars.pt"]
    models = [YOLO(name) for name in model_names]

    # Определяем цвета для каждой модели (при необходимости можно расширить список)
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # красный, синий, зеленый

    print("Выберите источник:")
    print("1 - видеофайл (test.mp4)")
    print("2 - одна веб-камера")
    print("3 - несколько веб-камер")
    print("4 - тест: одна веб-камера, 10 окон")
    choice = input("Введите номер источника: ").strip()

    # Инициализация источника(-ов)
    if choice == "1":
        caps = [cv2.VideoCapture("test.mp4")]
    elif choice == "2":
        caps = [cv2.VideoCapture(0)]
    elif choice == "3":
        indices = input("Введите индексы веб-камер через запятую (например, 0,1): ")
        try:
            cam_indices = [int(idx.strip()) for idx in indices.split(",")]
            caps = [cv2.VideoCapture(idx) for idx in cam_indices]
        except ValueError:
            print("Ошибка: неверный формат ввода индексов.")
            return
    elif choice == "4":
        # Для теста: одна веб-камера, 10 окон с дублированным кадром
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Ошибка: не удалось открыть веб-камеру.")
            return
    else:
        print("Неверный выбор источника.")
        return

    # Если не тестовый режим (варианты 1-3), проверяем открытие всех источников
    if choice in ["1", "2", "3"]:
        for cap in caps:
            if not cap.isOpened():
                print("Ошибка: не удалось открыть один или несколько источников видео.")
                return

    while True:
        if choice == "4":
            # Считываем кадр с единственной веб-камеры
            ret, frame = cap.read()
            if not ret:
                print("Ошибка: не удалось получить кадр с веб-камеры.")
                break
            # Создаем список из 10 одинаковых кадров
            frames = [frame] * 10
        else:
            frames = []
            # Чтение кадров с каждого источника
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    print("Ошибка: не удалось получить кадр с одного из источников.")
                    break
                frames.append(frame)
            if len(frames) != (1 if choice=="2" else len(caps)):
                break

        # Обработка и отображение каждого кадра
        for i, frame in enumerate(frames):
            annotated_frame = frame.copy()
            # Обрабатываем кадр каждой моделью
            for j, model in enumerate(models):
                results = model(frame)
                res = results[0]
                # Пример: для второй модели оставляем только боксы класса human (id = 0)
                filter_class = 0 if (j == 1) else None
                draw_boxes(annotated_frame, res, colors[j % len(colors)], model.names, filter_class=filter_class)

            if choice == "4":
                window_name = f"Test Webcam (Окно {i+1})"
            else:
                window_name = f"Combined YOLO Models (Cam {i})"
            cv2.imshow(window_name, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if choice == "4":
        cap.release()
    else:
        for cap in caps:
            cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
