import requests
import speech_recognition as sr
import pyttsx3

# Конфигурация
SERVER_URL = "http://192.168.1.100:8000/predict/"  # Замените на IP вашего сервера

# Инициализация TTS
engine = pyttsx3.init()

# Функция для преобразования текста в речь (TTS)
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Функция для преобразования речи в текст (STT)
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Слушаю...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        print("Распознаю...")
        text = recognizer.recognize_google(audio, language='ru-RU')
        print(f"Вы сказали: {text}")
        return text
    except sr.UnknownValueError:
        print("Не удалось распознать речь.")
        return None
    except sr.RequestError:
        print("Ошибка подключения к сервису распознавания речи.")
        return None

# Главная функция, которая организует обмен сообщениями
def chat_with_model():
    while True:
        # Получаем голосовой ввод
        user_input = listen()
        if user_input is None:
            continue

        if user_input.lower() == "выход":
            speak("Завершаю общение.")
            break

        # Отправляем запрос к нейросети
        response = requests.post(SERVER_URL, json={"question": user_input})
        if response.status_code == 200:
            model_response = response.json()["response"]
            print(f"Нейронка: {model_response}")
            speak(model_response)  # Преобразуем ответ в речь
        else:
            print("Ошибка:", response.status_code, response.text)
            speak("Извините, произошла ошибка.")

if __name__ == "__main__":
    chat_with_model()
