import json

# Путь к файлу
input_file_path = r"C:\Users\Vovas\OneDrive\Рабочий стол\Datasets\иии.txt"
output_file_path = "Вопрос-Ответ-ИИИ-Файл.json"

# Список для хранения пар "вопрос-ответ"
qa_pairs = []

# Открываем файл и читаем его построчно
with open(input_file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Инициализация переменных
current_question = None
current_answer = None

for line in lines:
    stripped_line = line.strip()
    if not stripped_line:
        continue  # Пропускаем пустые строки

    # Проверяем, начинается ли строка с "Вопрос" или "Ответ"
    if stripped_line.lower().startswith("вопрос"):
        if current_question and current_answer:
            # Сохраняем предыдущую пару, если она есть
            qa_pairs.append({"question": current_question, "answer": current_answer})
            current_answer = None
        current_question = stripped_line.split(":", 1)[-1].strip()  # Извлекаем текст после "Вопрос:"
    elif stripped_line.lower().startswith("ответ"):
        current_answer = stripped_line.split(":", 1)[-1].strip()  # Извлекаем текст после "Ответ:"
    else:
        # Пропускаем строки, которые не начинаются с "Вопрос" или "Ответ"
        continue

# Добавляем последнюю пару, если она существует
if current_question and current_answer:
    qa_pairs.append({"question": current_question, "answer": current_answer})

# Сохраняем результат в формате JSON
with open(output_file_path, "w", encoding="utf-8") as json_file:
    json.dump(qa_pairs, json_file, ensure_ascii=False, indent=4)

print(f"Данные сохранены в файл {output_file_path}")
