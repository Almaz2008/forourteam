import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import json
import os


def download_satellite_image(url, save_path="satellite_image.jpg"):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Изображение сохранено: {save_path}")
        else:
            print("Ошибка загрузки изображения!")
    except Exception as e:
        print(f"Ошибка: {e}")

# Пример URL (может потребоваться API-ключ)
# image_url = "https://www.nnvl.noaa.gov/images/satellite/globalir.jpg"
# download_satellite_image(image_url)

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Изображение не найдено!")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    return img, blurred



def detect_vortex_structure(image, min_radius=50, max_radius=300):
    circles = cv2.HoughCircles(
        image, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=100,
        param1=50, 
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f"Найдено {len(circles[0])} возможных циклонов!")
        return circles
    else:
        print("Циклоны не обнаружены.")
        return None


def fetch_weather_data(lat, lon, api_key="YOUR_API_KEY"):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = json.loads(response.text)
        pressure = data.get("main", {}).get("pressure", 0)  # гПа
        wind_speed = data.get("wind", {}).get("speed", 0)   # м/с
        return pressure, wind_speed
    else:
        print("Ошибка загрузки метеоданных!")
        return None, None

def is_tropical_cyclone(pressure, wind_speed):
    return pressure < 1000 and wind_speed > 17


def draw_cyclone_detection(original_img, circles, output_path="cyclone_detection.jpg"):
    img = original_img.copy()
    if circles is not None:
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            # Рисуем круг


cv2.circle(img, center, radius, (0, 255, 0), 4)
            cv2.circle(img, center, 2, (0, 0, 255), 3)
    
    cv2.imwrite(output_path, img)
    print(f"Результат сохранён: {output_path}")
    return img


def main():
    image_path = "satellite_image.jpg"
    if not os.path.exists(image_path):
        print("Скачиваем изображение...")
        download_satellite_image("https://www.nnvl.noaa.gov/images/satellite/globalir.jpg", image_path)
    
    original_img, processed_img = load_and_preprocess_image(image_path)
    

    circles = detect_vortex_structure(processed_img)
    
    if circles is not None:
        for i, circle in enumerate(circles[0, :]):
            lat, lon = circle[0], circle[1]  # Здесь нужно преобразование координат!
            pressure, wind_speed = fetch_weather_data(lat, lon)
            
            if pressure and wind_speed:
                if is_tropical_cyclone(pressure, wind_speed):
                    print(f"🌀 Обнаружен тропический циклон! Давление: {pressure} гПа, Ветер: {wind_speed} м/с")
                else:
                    print("❌ Не похоже на циклон (слишком слабые параметры).")
    
        result_img = draw_cyclone_detection(original_img, circles)
        
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title("Обнаружение тропического циклона")
        plt.axis("off")
        plt.show()
    else:
        print("Циклоны не найдены.")

if __name__ == "__main__":
    main()
