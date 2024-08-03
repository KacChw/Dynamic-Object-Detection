from ultralytics import YOLO
import cv2
import os

"""Program umożliwia korektę wykryć obiektów dynamicznych zrobioną przez model YOLO
    Legenda:
            lewy przycisk - zaznaczenie nowego obiektu
            prawy przycisk - usunięcie niepoprawnego obiektu
            n - następna klatka
            b - poprzednia klatka
            q - quit 
"""

model = YOLO('yolov8n.pt')

# zmienic na własne
frame_folder = 'recordings/result'
output_folder = 'recordings/cut'
#frame_folder = 'recordings/cala_trasa_front_frames'
#output_folder = 'recordings/result'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Funkcje do obsługi myszki
drawing = False  # True if mouse is pressed
ix, iy = -1, -1
rects = []
current_rect = []

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rects, current_rect

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        current_rect = [ix, iy, ix, iy]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_rect[2], current_rect[3] = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_rect[2], current_rect[3] = x, y
        rects.append(tuple(current_rect))
        current_rect = []

    elif event == cv2.EVENT_RBUTTONDOWN:
        remove_nearest_rectangle(x, y)

def remove_nearest_rectangle(x, y):
    global rects
    if rects:
        distances = [((rx1 + rx2) / 2 - x) ** 2 + ((ry1 + ry2) / 2 - y) ** 2 for rx1, ry1, rx2, ry2 in rects]
        min_index = distances.index(min(distances))
        rects.pop(min_index)

# Krok 3: Przetwarzanie klatek
frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith(('jpg', 'png', 'jpeg'))])
frame_index = 0

while True:
    frame_file = frame_files[frame_index]
    frame_path_input = os.path.join(frame_folder, frame_file)
    frame_path_output = os.path.join(output_folder, frame_file)

    if os.path.exists(frame_path_output):
        frame = cv2.imread(frame_path_output)
    else:
        frame = cv2.imread(frame_path_input)

    # Wykonaj detekcję i śledzenie obiektów tylko dla nowych klatek
    if not os.path.exists(frame_path_output):
        results = model.track(frame, persist=True)
        rects = []
        for result in results[0].boxes:
            rects.append(tuple(result.xyxy[0].tolist()))
    else:
        # Nie wykonuj detekcji, użyj zapamiętanych prostokątów
        rects = []  # Można tu zaimplementować logikę ładowania prostokątów z pliku, jeśli potrzeba

    # Ustaw interfejs myszki
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', draw_rectangle)

    while True:
        frame_copy = frame.copy()

        # Rysuj wszystkie prostokąty na klatce
        for rect in rects:
            cv2.rectangle(frame_copy, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (0, 255, 0), 2)

        # Rysuj aktualny prostokąt (jeśli rysowany)
        if drawing:
            cv2.rectangle(frame_copy, (current_rect[0], current_rect[1]), (current_rect[2], current_rect[3]),
                          (0, 0, 255), 2)

        cv2.imshow('frame', frame_copy)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):  # Przejdź do następnej klatki
            output_path = os.path.join(output_folder, frame_file)
            cv2.imwrite(output_path, frame_copy)
            frame_index = min(frame_index + 1, len(frame_files) - 1)
            break
        elif key == ord('b'):  # Cofnij do poprzedniej klatki
            output_path = os.path.join(output_folder, frame_file)
            cv2.imwrite(output_path, frame_copy)
            frame_index = max(frame_index - 1, 0)
            break
        elif key == ord('q'):  # Zakończ przetwarzanie
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
