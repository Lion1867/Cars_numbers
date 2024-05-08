import cv2
from imutils import contours
import easyocr
import streamlit as st
import numpy as np
import re
import os
import base64

# Create a temporary directory if it doesn't exist
TEMP_DIR = "temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Инициализация считывателя EasyOCR
reader = easyocr.Reader(['ru', 'en'])


# Картинка сверху
def add_top_image():
    with open("car_demo.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    top_image = f'''
    <style>
    .top-image {{
        background-image: url(data:image/jpg;base64,{encoded_string});
        background-size: cover;
        height: 200px;
        width: 100%;
        position: relative;
    }}
    </style>
    <div class="top-image"></div>
    '''
    st.markdown(top_image, unsafe_allow_html=True)


# Функция для добавления CSS-стилей
def add_css(css_code):
    st.markdown(f"<style>{css_code}</style>", unsafe_allow_html=True)


# Добавляем легкий оранжевый фон ко всему контенту на странице
add_css("""
    .main .block-container {
        background-color: #FFFFFF;
    }
    .main {
        background-color: #FFEFD5; /* легкий оранжевый */
    }
""")

# Заголовок приложения
add_top_image()
st.markdown("<h1 style='text-align: center; color: black;'>Распознавание номеров<br>автомобилей</h1>",
            unsafe_allow_html=True)

uploaded_file = st.file_uploader("Загрузить изображение или видео", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
    st.write(f"**Имя файла:** {file_details['FileName']}")
    st.write(f"**Тип файла:** {file_details['FileType']}")

    # Если файл - видео
    if uploaded_file.type.startswith('video'):
        # Сохраняем временный файл
        temp_video_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Создаем объект захвата видео
        video_capture = cv2.VideoCapture(temp_video_path)


        # Функция для обработки кадра видеопотока
        def process_frame(frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
            cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cnts, _ = contours.sort_contours(cnts)

            for c in cnts:
                area = cv2.contourArea(c)
                x, y, w, h = cv2.boundingRect(c)
                if area > 5000:
                    img = frame[y:y + h, x:x + w]
                    # Выполнение OCR с помощью EasyOCR
                    result = reader.readtext(img, detail=0, paragraph=False)
                    # Фильтрация результатов по длине
                    result = [re.sub('[/|\]\[]', '', text) for text in result]
                    # Выбор первого результата с длиной >= 8, предпочтительно более длинные результаты
                    result = [text for text in result if len(text) == 9] or [text for text in result if len(text) == 8]
                    if result:
                        st.success(f"Номер автомобиля: {result[0]}")
                        return True, cv2.cvtColor(frame,
                                                  cv2.COLOR_BGR2RGB)  # Номер найден, возвращаем True и обработанный кадр
            return False, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Номер не найден


        # Читаем кадр за кадром
        found_plate = False
        while not found_plate:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Обрабатываем каждый кадр
            found_plate, processed_frame = process_frame(frame)

            if found_plate:
                # Отображаем обработанный кадр
                st.image(processed_frame, caption='Обработанный кадр', use_column_width=True)
                break

        # Если номер не был найден
        if not found_plate:
            st.warning("Номер автомобиля не определен.")

        # Освобождаем ресурсы
        video_capture.release()


    # Если файл - изображение
    elif uploaded_file.type.startswith('image'):
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        height, width, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts, _ = contours.sort_contours(cnts)

        found_plate = False

        for c in cnts:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            if area > 5000:
                img = image[y:y + h, x:x + w]
                # Выполнение OCR с помощью EasyOCR
                result = reader.readtext(img, detail=0, paragraph=False)
                # Фильтрация результатов по длине
                result = [re.sub('[/|\]\[]', '', text) for text in result]
                # Выбор первого результата с длиной >= 8, предпочтительно более длинные результаты
                result = [text for text in result if len(text) == 9] or [text for text in result if len(text) == 8]
                if result:
                    st.success(f"Номер автомобиля: {result[0]}")
                    found_plate = True
                    break  # выводим только первый совпадающий результат и прекращаем обработку

        if not found_plate:
            st.warning("Номер автомобиля не определен.")

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Загруженное изображение.', use_column_width=True)