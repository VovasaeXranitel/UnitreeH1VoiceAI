import requests
import pyaudio
import wave
import io
import pygame
import logging
import speech_recognition as sr
import numpy as np
from scipy.signal import butter, lfilter
import sys

# Установка кодировки для вывода в консоль
sys.stdout.reconfigure(encoding='utf-8')

# Конфигурация
SERVER_URL = "http://192.168.1.100:8000/predict/"  # Замените на IP вашего сервера
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def remove_noise(audio_data, rate):
    # Конвертация байтов в numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Применение полосового фильтра для основных частот человеческого голоса (примерно 300-3400 Гц)
    filtered_audio = butter_bandpass_filter(audio_array, 300, 3400, rate, order=6)

    # Нормализация громкости
    max_value = np.max(np.abs(filtered_audio))
    if max_value > 0:
        filtered_audio = filtered_audio * (32767 / max_value) * 0.9

    # Конвертация обратно в байты
    filtered_audio = filtered_audio.astype(np.int16).tobytes()
    return filtered_audio

def list_audio_devices():
    p = pyaudio.PyAudio()
    print("\nДоступные устройства ввода:")
    input_devices = []
    output_devices = []

    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info.get('maxInputChannels') > 0:
            input_devices.append((i, dev_info.get('name')))
            print(f"{len(input_devices)-1}. {dev_info.get('name')} (ID: {i})")

    print("\nДоступные устройства вывода:")
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info.get('maxOutputChannels') > 0:
            output_devices.append((i, dev_info.get('name')))
            print(f"{len(output_devices)-1}. {dev_info.get('name')} (ID: {i})")
    p.terminate()
    return input_devices, output_devices

def record_audio(input_device_index, duration=5, rate=16000):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    p = pyaudio.PyAudio()
    logging.info(f"Запись с устройства ID: {input_device_index} на {duration} секунд...")
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=rate,
                        input=True,
                        input_device_index=input_device_index,
                        frames_per_buffer=CHUNK)

        frames = []
        for i in range(0, int(rate / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        # Сохраняем в буфер
        audio_data = b''.join(frames)

        # Очистка шума
        clean_audio = remove_noise(audio_data, rate)

        return clean_audio, rate
    except Exception as e:
        logging.error(f"Ошибка при записи звука: {e}")
        return None, None
    finally:
        p.terminate()

def play_audio(audio_bytes, output_device_index):
    try:
        pygame.mixer.quit()
        pygame.mixer.init(devicename=str(output_device_index))
        sound_file = io.BytesIO(audio_bytes)
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        logging.error(f"Ошибка воспроизведения звука: {e}")

def send_text_to_api(text):
    try:
        logging.info(f"Отправка текста на сервер: '{text}'")
        response = requests.post(SERVER_URL, json={"question": text})
        if response.status_code == 200:
            logging.info("Получен ответ от сервера")
            return response.content
        else:
            logging.error(f"Ошибка сервера: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logging.error(f"Ошибка при отправке запроса: {e}")
        return None

def chat_with_model(input_device_id, output_device_id):
    while True:
        print("\nВыберите действие:")
        print("1. Отправить текст")
        print("2. Отправить голосовой запрос")
        print("3. Выход")

        choice = input("Ваш выбор: ").strip()

        if choice == "1":
            user_text = input("Введите ваш запрос: ")
            if user_text.lower() in ['выход', 'exit', 'quit']:
                break

            audio_response = send_text_to_api(user_text)
            if audio_response:
                play_audio(audio_response, output_device_id)

        elif choice == "2":
            duration = int(input("Длительность записи (секунды): ") or "5")
            audio_data, rate = record_audio(input_device_id, duration)

            if audio_data:
                # Преобразование аудио в текст
                recognizer = sr.Recognizer()

                # Преобразование очищенного аудио в формат WAV
                wav_bytes = io.BytesIO()
                with wave.open(wav_bytes, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)  # 2 bytes for 16-bit audio
                    wav_file.setframerate(rate)
                    wav_file.writeframes(audio_data)
                wav_bytes.seek(0)
                with sr.AudioFile(wav_bytes) as source:
                    audio = recognizer.record(source)
                    try:
                        user_text = recognizer.recognize_google(audio, language="ru-RU")
                        print(f"Распознанный текст: {user_text}")

                        audio_response = send_text_to_api(user_text)
                        if audio_response:
                            play_audio(audio_response, output_device_id)
                    except sr.UnknownValueError:
                        print("Речь не распознана")
                    except sr.RequestError as e:
                        print(f"Ошибка сервиса распознавания: {e}")

        elif choice == "3":
            break

        else:
            print("Неверный выбор. Попробуйте снова.")

if __name__ == "__main__":
    print("Инициализация аудиоустройств...")
    input_devices, output_devices = list_audio_devices()

    if not input_devices:
        logging.error("Не найдены устройства ввода звука!")
        exit(1)
    if not output_devices:
        logging.error("Не найдены устройства вывода звука!")
        exit(1)

    input_choice = int(input("\nВыберите устройство ввода (номер): ") or "0")
    output_choice = int(input("Выберите устройство вывода (номер): ") or "0")

    if 0 <= input_choice < len(input_devices) and 0 <= output_choice < len(output_devices):
        input_device_id = input_devices[input_choice][0]
        output_device_id = output_devices[output_choice][0]

        logging.info(f"Выбрано устройство ввода: {input_devices[input_choice][1]}")
        logging.info(f"Выбрано устройство вывода: {output_devices[output_choice][1]}")

        chat_with_model(input_device_id, output_device_id)
    else:
        logging.error("Неверный выбор устройства!")
