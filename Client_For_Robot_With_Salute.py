import requests
import sounddevice as sd
import io
import scipy.io.wavfile as wav
import json

with open("config.json", encoding="utf-8") as config_file:
    config = json.load(config_file)

# Конфигурация авторизации
GIGACHAT_AUTH_KEY = config["GIGACHAT_API_KEY"]
SBER_SALUTE_AUTH_KEY = config["SALUTESPEECH_API_KEY"]
SAMPLE_RATE = 16000

# Вывод списка устройств
def list_audio_devices():
    devices = sd.query_devices()
    input_devices = []
    output_devices = []

    print("\nУстройства ВВОДА:")
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            input_devices.append((i, dev['name']))
            print(f"{len(input_devices)-1}. {dev['name']} (ID: {i})")

    print("\nУстройства ВЫВОДА:")
    for i, dev in enumerate(devices):
        if dev['max_output_channels'] > 0:
            output_devices.append((i, dev['name']))
            print(f"{len(output_devices)-1}. {dev['name']} (ID: {i})")

    return input_devices, output_devices

# Выбор устройств
def select_devices():
    inputs, outputs = list_audio_devices()
    input_idx = int(input("\nВыберите номер устройства ВВОДА: "))
    output_idx = int(input("Выберите номер устройства ВЫВОДА: "))
    return inputs[input_idx][0], outputs[output_idx][0]

# Запись аудио
def record_audio(duration, device):
    print("Говорите...")
    audio = sd.rec(int(duration*SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, device=device, dtype='int16')
    sd.wait()
    print("Запись завершена.")
    return audio.flatten()

# SaluteSpeech STT
def transcribe_audio(audio_bytes, salute_token):
    url = "https://smartspeech.sber.ru/rest/v1/speech:recognize"
    headers = {'Authorization': f'Bearer {salute_token}',
               "Content-Type": "audio/x-pcm;bit=16;rate=16000"}
    response = requests.post(url, headers=headers, data=audio_bytes, verify=False)
    if response.status_code == 200:
        return response.json()['result'][0]
    print("STT:", response.text)
    return "речь не распознана, повторите"

# Получение токена GigaChat
def get_gigachat_token():
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    payload = {'scope': 'GIGACHAT_API_PERS'}
    headers = {
        'Authorization': f'Basic {GIGACHAT_AUTH_KEY}',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': '5c107b81-fa83-408d-a7ff-426eec35e0e8'
    }

    try:
        response = requests.post(url, headers=headers, data=payload, verify=False)

        # Проверим статус ответа
        if response.ok:
            response_json = response.json()
            print("Токен успешно получен.")
            return response_json.get("access_token")
        else:
            print(f"Ошибка запроса токена. Статус: {response.status_code}, Ответ: {response.text}")
            return None

    except requests.exceptions.JSONDecodeError as e:
        print(f"Ошибка JSON-декодирования. Текст ответа: {response.text}")
        return None
    except requests.RequestException as e:
        print(f"❌ Сетевая ошибка: {e}")
        return None

def get_salute_token():
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    payload = {'scope': 'SALUTE_SPEECH_PERS'}
    headers = {
        'Authorization': f'Basic {SBER_SALUTE_AUTH_KEY}',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': '5c107b81-fa83-408d-a7ff-426eec35e0e8'
    }

    response = requests.post(url, headers=headers, data=payload, verify=False)

    if response.ok:
        token = response.json().get("access_token")
        print("SaluteSpeech токен успешно получен.")
        return token
    else:
        print(f"Ошибка получения токена SaluteSpeech: {response.text}")
        return None

# Запрос к GigaChat
def ask_gigachat(question, token):
    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }
    data = {
        "model": "GigaChat-Pro",
        "messages": [{"role": "user", "content": question}]
    }
    response = requests.post(url, headers=headers, json=data, verify=False)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    print("GigaChat:", response.text)
    return "Ошибка генерации ответа."

# SaluteSpeech TTS
def synthesize_speech(text, salute_token):
    url = "https://smartspeech.sber.ru/rest/v1/text:synthesize"
    headers = {
        "Authorization": f"Bearer {salute_token}",
        "Content-Type": "application/text",
        "Accept": "audio/wav"
    }
    response = requests.post(url, headers=headers, data=text.encode('utf-8'), verify=False)
    if response.status_code == 200:
        return response.content
    print("TTS:", response.text)
    return None

# Воспроизведение аудио
def play_audio(audio_bytes, device):
    try:
        rate, audio_array = wav.read(io.BytesIO(audio_bytes))
        sd.play(audio_array, rate, device=device)
        sd.wait()
    except Exception as e:
        print(f"Ошибка воспроизведения: {e}")

# Основной цикл общения
def main():
    input_device, output_device = select_devices()

    # Получаем токены
    giga_token = get_gigachat_token()
    salute_token = get_salute_token()

    # Проверка обоих токенов!
    if not giga_token or not salute_token:
        print("Не удалось получить один или оба токена (GigaChat/SaluteSpeech).")
        return

    while True:
        audio = record_audio(5, input_device)
        text = transcribe_audio(audio.tobytes(), salute_token)
        print("Вы сказали:", text)

        if "выход" in text.lower():
            print("Завершение работы.")
            break

        answer = ask_gigachat(text, giga_token)
        print("GigaChat:", answer)

        audio_reply = synthesize_speech(answer, salute_token)
        if audio_reply:
            play_audio(audio_reply, output_device)
        else:
            print("Ошибка TTS, ответ:", answer)

if __name__ == "__main__":
    main()
