import requests

SERVER_URL = "http://192.168.1.100:8000/predict/"  # Замените на IP сервера

while True:
    question = input("Вы: ")
    if question.lower() == "exit":
        break

    response = requests.post(SERVER_URL, json={"question": question})
    if response.status_code == 200:
        print(f"Нейронка: {response.json()['response']}")
    else:
        print("Ошибка:", response.status_code, response.text)
