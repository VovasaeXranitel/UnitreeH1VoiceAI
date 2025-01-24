import json
from charset_normalizer import detect

# Путь к файлу
input_file_path = r"C:\Users\Vovas\OneDrive\Рабочий стол\Datasets\Данные о СНО ИИИ для нейронки.txt"
output_file_path = "Данные-СНО-ИИИ-Файл.json"

# Список для хранения абзацев
paragraphs = []

# Определение кодировки файла
with open(input_file_path, "rb") as file:
    raw_data = file.read()
    detected = detect(raw_data)
    file_encoding = detected["encoding"]

# Открываем файл с определенной кодировкой и читаем его построчно
with open(input_file_path, "r", encoding=file_encoding) as file:
    lines = file.readlines()

# Переменные для хранения текущего абзаца
current_paragraph = []

for line in lines:
    stripped_line = line.strip()

    # Если строка пустая, то это разделитель между абзацами
    if not stripped_line:
        if current_paragraph:
            paragraph = " ".join(current_paragraph).strip()
            paragraphs.append(paragraph)
            current_paragraph = []
        continue

    # Добавляем строку в текущий абзац
    current_paragraph.append(stripped_line)

# Обрабатываем последний абзац, если он существует
if current_paragraph:
    paragraph = " ".join(current_paragraph).strip()
    paragraphs.append(paragraph)

# Сохраняем результат в формате JSON
with open(output_file_path, "w", encoding="utf-8") as json_file:
    json.dump(paragraphs, json_file, ensure_ascii=False, indent=4)

print(f"Данные сохранены в файл {output_file_path}")