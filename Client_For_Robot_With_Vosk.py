import requests
import sounddevice as sd
import scipy.io.wavfile as wav
import io
import json
from vosk import Model, KaldiRecognizer

# Конфигурация
SERVER_URL = "http://192.168.1.100:8000/predict/"
SAMPLE_RATE = 16000
VOSK_MODEL_PATH = r"C:\Users\Vovas\Downloads\Vosk\vosk-model-small-ru-0.22"
model = Model(VOSK_MODEL_PATH)

# Чистый список устройств ввода и вывода
def list_audio_devices():
    devices = sd.query_devices()

    input_devices = []
    output_devices = []
    input_names_seen = set()
    output_names_seen = set()

    for idx, device in enumerate(devices):
        name = device['name']
        if device['max_input_channels'] > 0 and name not in input_names_seen:
            input_devices.append((idx, name))
            input_names_seen.add(name)
        if device['max_output_channels'] > 0 and name not in output_names_seen:
            output_devices.append((idx, name))
            output_names_seen.add(name)

    print("\nДоступные устройства ВВОДА:")
    for idx, (id_, name) in enumerate(input_devices):
        print(f"{idx}: {name} (ID устройства: {id_})")

    print("\nДоступные устройства ВЫВОДА:")
    for idx, (id_, name) in enumerate(output_devices):
        print(f"{idx}: {name} (ID устройства: {id_})")

    return input_devices, output_devices

# Выбор устройств из списка
def select_devices():
    input_devices, output_devices = list_audio_devices()

    while True:
        try:
            input_choice = int(input("Выберите номер устройства ВВОДА: "))
            if 0 <= input_choice < len(input_devices):
                input_id = input_devices[input_choice][0]
            else:
                print("Некорректный выбор устройства ввода.")
                continue

            output_choice = int(input("Выберите номер устройства ВЫВОДА: "))
            if 0 <= output_choice < len(output_devices):
                output_id = output_devices[output_choice][0]
            else:
                print("Некорректный выбор устройства вывода.")
                continue

            return input_id, output_id
        except ValueError:
            print("Введите числовой номер устройства.")

# Запись аудио
def record_audio(duration=30, device=None):
    try:
        audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=1, dtype='int16', device=device)
        sd.wait()
        return audio.flatten()
    except sd.PortAudioError as e:
        print(f"Ошибка записи: {e}")
        return None

# STT (Vosk)
def vosk_stt(audio_data):
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(True)

    if recognizer.AcceptWaveform(audio_data.tobytes()):
        result = json.loads(recognizer.Result())
    else:
        result = json.loads(recognizer.FinalResult())

    text = result.get('text', '').strip()
    print(f"Распознано: {text}")
    return text if text else None

# Воспроизведение аудио
def play_audio(audio_data, sample_rate, output_device=None):
    try:
        sd.play(audio_data, sample_rate, device=output_device, blocksize=4096)
        sd.wait()
    except Exception as e:
        print(f"Ошибка воспроизведения: {e}")

# Основной цикл общения
def chat_with_model():
    input_device, output_device = select_devices()
    print("Начинайте говорить. Для выхода скажите 'выход'.")

    while True:
        audio_input = record_audio(duration=5, device=input_device)

        if audio_input is None:
            print("Ошибка записи. Повторите выбор устройств.")
            input_device, output_device = select_devices()
            continue

        user_text = vosk_stt(audio_input)

        if not user_text:
            print("Не удалось распознать. Повторите.")
            continue

        if "выход" in user_text.lower():
            print("Выход.")
            break

        try:
            response = requests.post(
                SERVER_URL,
                json={"question": user_text},
                headers={"Accept": "audio/wav"},
                timeout=15
            )

            if response.status_code == 200:
                if 'audio' in response.headers.get('Content-Type', ''):
                    audio_data, fs = wav.read(io.BytesIO(response.content))
                    print("Воспроизвожу ответ...")
                    play_audio(audio_data, fs, output_device=output_device)
                else:
                    text_response = response.json().get("response", "Нет ответа")
                    print(f"Ответ текстом: {text_response}")
            else:
                print(f"Ошибка сервера: {response.status_code} {response.text}")
        except Exception as e:
            print(f"Ошибка запроса: {e}")

if __name__ == "__main__":
    chat_with_model()
