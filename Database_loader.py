import json
import pyodbc

# === Конфигурация подключения к MSSQL ===
SERVER = r"VLADIMIR_LAPTOP\SQLEXPRESS"  # Имя сервера MSSQL
DATABASE = r"Mirea_Projects"  # Имя базы данных
TABLE_NAME = r"qa_data"  # Название таблицы

# === Подключение к MSSQL ===
conn = pyodbc.connect(
    f"DRIVER={{SQL Server}};SERVER={SERVER};DATABASE={DATABASE};"
)
cursor = conn.cursor()

# === Создание таблицы, если её нет ===
cursor.execute(f"""
    IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{TABLE_NAME}')
    CREATE TABLE {TABLE_NAME} (
        id INT IDENTITY(1,1) PRIMARY KEY,
        question NVARCHAR(MAX),
        answer NVARCHAR(MAX)
    )
""")
conn.commit()

# === Загрузка JSON-файла ===
JSON_FILE_PATH = r"C:\UnitreeH1VoiceAI\Полный_Датасет.json"  # Укажите путь к файлу JSON

with open(JSON_FILE_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

# === Загрузка данных в таблицу ===
for item in data:
    cursor.execute(f"""
        INSERT INTO {TABLE_NAME} (question, answer) VALUES (?, ?)
    """, item["question"], item["answer"])

conn.commit()
cursor.close()
conn.close()

print("Данные успешно загружены в MSSQL!")
