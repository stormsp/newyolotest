from ultralytics import YOLO
import cv2
from datetime import datetime

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
            if filter_class is not None and int(cls) != filter_class:
                continue
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            class_name = class_names.get(int(cls), str(int(cls)))
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    # Список имён файлов моделей (весов)
    model_names = ["fire.pt", "person.pt", "gun.pt", "flood.pt", "tree.pt", "cars.pt"]
    models = [YOLO(name) for name in model_names]

    # Определяем цвета для каждой модели (красный, синий, зеленый)
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]

    print("Выберите источник:")
    print("1 - видеофайл (test.mp4)")
    print("2 - одна веб-камера")
    print("3 - несколько веб-камер")
    print("4 - тест: одна веб-камера, 10 окон")
    choice = input("Введите номер источника: ").strip()

    active_cam_indices = []
    if choice == "1":
        caps = [cv2.VideoCapture("test.mp4")]
        active_cam_indices = [0]
    elif choice == "2":
        caps = [cv2.VideoCapture(0)]
        active_cam_indices = [0]
    elif choice == "3":
        indices = input("Введите индексы веб-камер через запятую (например, 0,1): ")
        try:
            cam_indices = [int(idx.strip()) for idx in indices.split(",")]
            caps = [cv2.VideoCapture(idx) for idx in cam_indices]
            active_cam_indices = cam_indices[:]  # копия списка индексов
        except ValueError:
            print("Ошибка: неверный формат ввода индексов.")
            return
    elif choice == "4":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Ошибка: не удалось открыть веб-камеру.")
            return
        active_cam_indices = [0]
    else:
        print("Неверный выбор источника.")
        return

    if choice in ["1", "2", "3"]:
        for cap in caps:
            if not cap.isOpened():
                print("Ошибка: не удалось открыть один или несколько источников видео.")
                return

    # ------------------ Конфигурация контроля нарушений (Intruder) ------------------
    # Если переменная intrusion_check равна True, то для камер с указанными индексами производится проверка
    intrusion_check = True         # True - проверка выполняется, False - проверка отключена
    intrusion_cam_indices = [0]    # Список индексов камер, для которых выполняется проверка
    allowed_time_str = "08:00-20:00" # Разрешённое время в формате HH:MM-HH:MM
    start_str, end_str = allowed_time_str.split("-")
    start_time = datetime.strptime(start_str.strip(), "%H:%M").time()
    end_time = datetime.strptime(end_str.strip(), "%H:%M").time()
    # ----------------------------------------------------------------------------------

    while True:
        if choice == "4":
            ret, frame = cap.read()
            if not ret:
                print("Ошибка: не удалось получить кадр с веб-камеры.")
                break
            frames = [frame] * 10
        else:
            frames = []
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    print("Ошибка: не удалось получить кадр с одного из источников.")
                    break
                frames.append(frame)
            if len(frames) != (1 if choice == "2" else len(caps)):
                break

        for i, frame in enumerate(frames):
            annotated_frame = frame.copy()
            person_count = 0

            for j, model in enumerate(models):
                results = model(frame)
                res = results[0]

                # Для модели person.pt (предполагаем, что она идёт с индексом 1) фильтруем класс person (id = 0)
                if j == 1:
                    filter_class = 0
                    if res.boxes is not None and len(res.boxes) > 0:
                        try:
                            classes = res.boxes.cls.cpu().numpy()
                        except AttributeError:
                            classes = res.boxes.cls
                        person_count = sum(1 for cls in classes if int(cls) == 0)
                else:
                    filter_class = None

                draw_boxes(annotated_frame, res, colors[j % len(colors)], model.names, filter_class=filter_class)

            # Если обнаружено более 10 человек, выводим сообщение "crowd"
            if person_count > 10:
                cv2.putText(annotated_frame, "crowd", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            # Если включена проверка контроля нарушений и текущая камера входит в список
            if intrusion_check and active_cam_indices[i] in intrusion_cam_indices:
                current_time = datetime.now().time()
                if start_time <= end_time:
                    allowed = start_time <= current_time <= end_time
                else:
                    allowed = current_time >= start_time or current_time <= end_time

                if not allowed and person_count > 0:
                    cv2.putText(annotated_frame, "Intruder", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            if choice == "4":
                window_name = f"Test Webcam (Окно {i+1})"
            else:
                window_name = f"Combined YOLO Models (Cam {active_cam_indices[i]})"
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
