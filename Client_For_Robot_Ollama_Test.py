import requests
import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile as wav
import io
import os
import subprocess

# Конфигурация
STT_MODEL = "dimavz/whisper-tiny:latest"
SERVER_URL = "http://192.168.1.100:8000/predict/"
SAMPLE_RATE = 16000  # Whisper рекомендует 16 kHz

# Запись аудио с микрофона
def record_audio(duration=5, device=None):
    print("Запись аудио...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype='float32', device=device)
    sd.wait()
    print("Запись завершена.")
    return audio.flatten()

# STT через Ollama Whisper (CLI)
def whisper_stt(audio_data):
    # Сохраняем аудио во временный файл
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        wav.write(temp_wav.name, SAMPLE_RATE, np.int16(audio_data * 32767))

    try:
        # Выполняем команду через CLI Ollama
        result = subprocess.run(
            ["ollama", "run", STT_MODEL, temp_wav.name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode == 0:
            text = result.stdout.strip()
            print(f"Распознано: {text}")
            return text if text else None
        else:
            print(f"Ошибка Ollama Whisper: {result.stderr.strip()}")
            return None
    finally:
        os.unlink(temp_wav.name)

# Воспроизведение аудио-ответа
def play_audio(audio_data, sample_rate, output_device=None):
    sd.play(audio_data, sample_rate, device=output_device)
    sd.wait()

# Основная функция общения
def chat_with_model():
    input_device = int(input("Введите ID устройства ввода: "))
    output_device = int(input("Введите ID устройства вывода: "))

    print("Начинайте говорить. Для завершения скажите 'выход'.")

    while True:
        audio_input = record_audio(duration=5, device=input_device)
        user_text = whisper_stt(audio_input)

        if user_text is None:
            print("Не удалось распознать. Повторите.")
            continue

        if "выход" in user_text.lower():
            print("Завершение работы.")
            break

        try:
            response = requests.post(
                SERVER_URL,
                json={"question": user_text},
                headers={"Accept": "audio/wav"},
                timeout=15
            )

            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                if 'audio' in content_type:
                    audio_data, fs = wav.read(io.BytesIO(response.content))
                    print("Воспроизвожу ответ...")
                    play_audio(audio_data, fs, output_device=output_device)
                else:
                    try:
                        text_response = response.json().get("response", "")
                        print(f"Ответ текстом: {text_response}")
                    except:
                        print("Неизвестный формат ответа.")
            else:
                print(f"Ошибка связи с сервером: {response.status_code} {response.text}")
        except requests.RequestException as e:
            print(f"Ошибка при отправке запроса: {e}")

if __name__ == "__main__":
    chat_with_model()
