import requests
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os
import io
import time
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from soundcloud_player import SoundCloudPlayer

# Конфигурация
SERVER_URL = "http://192.168.1.100:8000/predict/"  # Замените на IP вашего сервера
SAMPLE_RATE = 44100  # Частота дискретизации
sc_player = SoundCloudPlayer()


# Функции для фильтрации шума
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


def denoise_audio(audio_data, fs, lowcut=300, highcut=3000):
    # Полосовой фильтр для выделения частот человеческого голоса (примерно 300-3000 Гц)
    filtered_audio = butter_bandpass_filter(audio_data, lowcut, highcut, fs)

    # Нормализация аудио
    max_val = np.max(np.abs(filtered_audio))
    if max_val > 0:
        normalized_audio = filtered_audio / max_val * 0.9  # 90% от максимальной амплитуды
    else:
        normalized_audio = filtered_audio

    # Шумоподавление с помощью порогового значения
    threshold = 0.05  # Порог шума (настраиваемый параметр)
    noise_gate = np.where(np.abs(normalized_audio) < threshold, 0, normalized_audio)

    return noise_gate


# Функция для вывода списка аудиоустройств с улучшенной информацией
def list_audio_devices():
    # Обновляем список устройств (важно для Bluetooth-устройств)
    sd.check_output_settings()
    sd.check_input_settings()

    # Получаем обновленный список всех устройств
    devices = sd.query_devices()

    print("\nДоступные устройства ввода:")
    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            device_type = "Bluetooth" if "bluetooth" in device['name'].lower() else "Проводное"
            input_devices.append((i, device['name'], device_type))
            print(f"{len(input_devices) - 1}. {device['name']} (ID: {i}, Тип: {device_type})")

    print("\nДоступные устройства вывода:")
    output_devices = []
    for i, device in enumerate(devices):
        if device['max_output_channels'] > 0:
            device_type = "Bluetooth" if "bluetooth" in device['name'].lower() else "Проводное"
            output_devices.append((i, device['name'], device_type))
            print(f"{len(output_devices) - 1}. {device['name']} (ID: {i}, Тип: {device_type})")

    return input_devices, output_devices


# Функция для выбора устройств
def select_devices():
    print("Сканирование аудиоустройств...")
    # Делаем небольшую паузу, чтобы Bluetooth-устройства успели инициализироваться
    time.sleep(1)

    input_devices, output_devices = list_audio_devices()
    # Выбор устройства ввода
    input_id = None
    while input_id is None:
        try:
            input_choice = int(input("\nВыберите номер устройства ввода: "))
            if 0 <= input_choice < len(input_devices):
                input_id = input_devices[input_choice][0]
                print(
                    f"Выбрано устройство ввода: {input_devices[input_choice][1]} (Тип: {input_devices[input_choice][2]})")
            else:
                print("Некорректный выбор. Пожалуйста, выберите из списка.")
        except ValueError:
            print("Введите числовое значение.")

    # Выбор устройства вывода
    output_id = None
    while output_id is None:
        try:
            output_choice = int(input("Выберите номер устройства вывода: "))
            if 0 <= output_choice < len(output_devices):
                output_id = output_devices[output_choice][0]
                print(
                    f"Выбрано устройство вывода: {output_devices[output_choice][1]} (Тип: {output_devices[output_choice][2]})")
            else:
                print("Некорректный выбор. Пожалуйста, выберите из списка.")
        except ValueError:
            print("Введите числовое значение.")

    return input_id, output_id


# Функция для воспроизведения аудио с учетом специфики Bluetooth-устройств
def play_audio(audio_data, sample_rate, output_device=None):
    try:
        # Получаем информацию о выбранном устройстве
        device_info = sd.query_devices(output_device)
        device_name = device_info['name']

        # Проверяем, является ли устройство Bluetooth
        is_bluetooth = "bluetooth" in device_name.lower()

        if is_bluetooth:
            print(f"Воспроизведение через Bluetooth-устройство: {device_name}")
            # Для Bluetooth устройств иногда нужна небольшая пауза перед воспроизведением
            time.sleep(0.5)

            # Конвертируем аудио в 16-бит (часто более стабильно для Bluetooth)
            if audio_data.dtype != np.int16:
                if audio_data.dtype == np.float32:
                    audio_data = np.int16(audio_data * 32767)
                else:
                    audio_data = audio_data.astype(np.int16)

        # Устанавливаем устройство вывода как текущее
        sd.default.device = output_device

        # Воспроизведение аудио через sounddevice с увеличенным буфером для Bluetooth
        blocksize = 4096 if is_bluetooth else 1024

        sd.play(audio_data, sample_rate, device=output_device, blocksize=blocksize)
        sd.wait()  # Ждем окончания воспроизведения

        # Добавляем небольшую паузу после воспроизведения для Bluetooth-устройств
        if is_bluetooth:
            time.sleep(0.5)
    except Exception as e:
        print(f"Ошибка воспроизведения аудио: {e}")
        print("Пробую альтернативный метод воспроизведения...")

        try:
            # Альтернативный метод воспроизведения
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_filename = temp_file.name
            temp_file.close()

            # Сохраняем аудио во временный файл
            wavfile.write(temp_filename, sample_rate, audio_data)

            # Используем os.system для воспроизведения через системный плеер
            if os.name == 'nt':  # Windows
                os.system(f'start {temp_filename}')
            elif os.name == 'posix':  # Linux/Mac
                os.system(f'aplay {temp_filename}')

            # Даем время на воспроизведение
            time.sleep(5)

            # Удаляем временный файл
            try:
                os.unlink(temp_filename)
            except:
                pass

        except Exception as e2:
            print(f"Альтернативный метод тоже не сработал: {e2}")


# Функция для преобразования речи в текст (STT) с использованием sounddevice
def listen(input_device=None, duration=5):
    print("Слушаю...")

    try:
        # Получаем информацию о выбранном устройстве
        device_info = sd.query_devices(input_device)
        device_name = device_info['name']

        # Проверяем, является ли устройство Bluetooth
        is_bluetooth = "bluetooth" in device_name.lower()

        # Параметры записи зависят от типа устройства
        channels = 1  # Mono запись
        blocksize = 4096 if is_bluetooth else 1024

        if is_bluetooth:
            print(f"Запись с Bluetooth-микрофона: {device_name}")
            # Для Bluetooth устройств может потребоваться пауза перед записью
            time.sleep(0.5)

        # Записываем аудио с выбранного устройства
        recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                           channels=channels, dtype='float32', device=input_device,
                           blocksize=blocksize)

        print("Запись...")
        sd.wait()  # Ждем окончания записи
        print("Запись завершена. Обработка аудио...")

        # Применяем шумоподавление и фильтрацию
        recording = recording.flatten()  # Преобразуем в одномерный массив, если запись многоканальная
        cleaned_recording = denoise_audio(recording, SAMPLE_RATE)
        print("Аудио очищено от шумов")

        # Создаем временный WAV файл для speech_recognition
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name

        # Нормализуем аудио и сохраняем как WAV
        cleaned_recording = np.int16(cleaned_recording * 32767)
        wav.write(temp_filename, SAMPLE_RATE, cleaned_recording)

        # Используем speech_recognition для распознавания
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_filename) as source:
            audio = recognizer.record(source)
            # Дополнительная настройка для шумоподавления в speech_recognition
            recognizer.dynamic_energy_threshold = True
            recognizer.energy_threshold = 300  # Настраиваемый порог энергии

        # Удаляем временный файл
        try:
            os.unlink(temp_filename)
        except:
            pass

        try:
            print("Распознаю...")
            text = recognizer.recognize_google(audio, language='ru-RU')
            print(f"Вы сказали: {text}")
            return text
        except sr.UnknownValueError:
            print("Не удалось распознать речь.")
            return None
        except sr.RequestError as e:
            print(f"Ошибка подключения к сервису распознавания речи: {e}")
            return None
    except Exception as e:
        print(f"Ошибка при записи аудио: {e}")
        return None


# Функция для обновления списка устройств
def refresh_devices():
    print("Обновление списка устройств...")
    # Сбрасываем кэш аудиоустройств
    sd._terminate()
    sd._initialize()
    time.sleep(1)  # Даем время на инициализацию
    return list_audio_devices()


# Главная функция, которая организует обмен сообщениями
def chat_with_model():
    # Выбор устройств
    input_device, output_device = select_devices()
    print(f"Выбрано устройство ввода ID: {input_device}")
    print(f"Выбрано устройство вывода ID: {output_device}")

    print("\nСистема готова к работе.")
    print("Команды:")
    print("- 'обновить устройства' - обновить список аудиоустройств")
    print("- 'выход' - завершить программу")
    print("\nНачните говорить...")

    while True:
        # Получаем голосовой ввод
        user_input = listen(input_device=input_device)
        if user_input is None:
            print("Попробуйте еще раз или скажите 'обновить устройства' для обновления списка.")
            continue

        # Обработка специальных команд
        if user_input.lower() == "выход":
            print("Завершаю общение.")
            break
        elif user_input.lower() in ["обновить устройства", "сменить устройства", "обновить"]:
            input_devices, output_devices = refresh_devices()
            input_device, output_device = select_devices()
            print(f"Выбрано устройство ввода ID: {input_device}")
            print(f"Выбрано устройство вывода ID: {output_device}")
            continue
        elif user_input.lower().startswith("soundcloud"):
            query = user_input.partition(" ")[2].strip()
            if not query:
                print("Укажите запрос для SoundCloud")
            else:
                try:
                    sc_player.play(query, output_device=output_device)
                except Exception as e:
                    print(f"Ошибка SoundCloud: {e}")
            continue

        # Отправляем запрос к нейросети
        try:
            print("Отправка запроса на сервер...")
            # Отправляем запрос и указываем, что ожидаем аудио в ответе
            response = requests.post(
                SERVER_URL,
                json={"question": user_input},
                headers={"Accept": "audio/wav"}  # Запрашиваем аудио формат
            )

            if response.status_code == 200:
                # Проверяем тип контента в ответе
                content_type = response.headers.get('Content-Type', '')

                if 'audio' in content_type:
                    # Получаем аудио ответ
                    audio_data = io.BytesIO(response.content)

                    # Преобразуем аудио-файл в numpy массив
                    sample_rate, audio_array = wavfile.read(audio_data)

                    # Воспроизводим аудио
                    print("Воспроизвожу ответ...")
                    play_audio(audio_array, sample_rate, output_device)
                else:
                    # Если пришел текстовый ответ, а не аудио
                    try:
                        model_response = response.json().get("response", "Нет ответа")
                        print(f"Текстовый ответ: {model_response}")
                    except:
                        print("Получен неизвестный формат ответа")
            else:
                print("Ошибка:", response.status_code, response.text)
        except Exception as e:
            print(f"Ошибка при отправке запроса: {e}")


if __name__ == "__main__":
    chat_with_model()
