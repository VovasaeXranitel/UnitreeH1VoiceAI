import torch
import pyodbc
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === Проверка доступности GPU ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# === Конфигурация MSSQL ===
SERVER = "your_server_name"  # Имя сервера MSSQL
DATABASE = "your_database_name"  # Имя базы данных
USERNAME = "your_username"  # Логин
PASSWORD = "your_password"  # Пароль
TABLE_NAME = "QuestionAnswer"  # Название таблицы с вопросами и ответами

# === Конфигурация модели ===
TOKENIZER_PATH = "/home/vremennaya-kpu/.cache/huggingface/hub/models--ai-forever--rugpt3small_based_on_gpt2/rugpt3_tokenizer"
MODEL_PATH = "/home/vremennaya-kpu/.cache/huggingface/hub/models--ai-forever--rugpt3small_based_on_gpt2"
OUTPUT_DIR = "/home/vremennaya-kpu/.cache/huggingface/hub/Output_Fine_Tuned_Model_RUGPT_v2"

# === Функция загрузки данных из БД ===
def load_data_from_db():
    print("Подключение к базе данных MSSQL...")
    conn = pyodbc.connect(
        f"DRIVER={{SQL Server}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD}"
    )
    cursor = conn.cursor()

    # Загружаем данные из таблицы
    print("Загрузка данных из БД...")
    cursor.execute(f"SELECT question, answer FROM {TABLE_NAME}")
    data = cursor.fetchall()

    # Закрываем соединение
    cursor.close()
    conn.close()

    # Форматируем данные в список словарей
    formatted_data = [{"question": row[0], "answer": row[1]} for row in data]
    return formatted_data

# === Функция предобработки данных ===
def preprocess_function(examples, tokenizer):
    question = str(examples["question"]) if examples["question"] else ""
    answer = str(examples["answer"]) if examples["answer"] else ""

    inputs = tokenizer(
        question,
        text_pair=answer,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

# === Функция расчёта метрик ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits).to(device), dim=-1)
    labels = torch.tensor(labels).to(device)

    accuracy = accuracy_score(labels.cpu().numpy().flatten(), predictions.cpu().numpy().flatten())
    return {"accuracy": accuracy}

# === Основная функция ===
def train_model():
    # Загрузка данных
    data_list = load_data_from_db()

    # Разделение данных на обучающую и валидационную выборки
    print("Разделение данных...")
    train_data_list, val_data_list = train_test_split(data_list, test_size=0.1, random_state=42)
    train_data = Dataset.from_list(train_data_list)
    val_data = Dataset.from_list(val_data_list)

    # Проверка примеров данных
    print("Пример данных перед токенизацией:")
    print(train_data[:5])

    # Подготовка токенизатора
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # Токенизация данных
    print("Токенизация данных...")
    tokenized_train_data = train_data.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    tokenized_val_data = val_data.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # Загрузка модели на GPU
    print("Загрузка модели на GPU...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)

    # Настройка параметров обучения
    print("Настройка тренировки...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=2e-5,  # Оптимальный LR для LLM
        per_device_train_batch_size=4,  # Уменьшаем для видеокарты
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
        fp16=True  # Включаем mixed precision для ускорения на GPU
    )

    # Инициализация Trainer
    print("Инициализация Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_val_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Обучение модели
    print("Начало обучения...")
    trainer.train()

    # Сохранение модели
    print("Сохранение модели...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Оценка модели
    print("Оценка модели...")
    eval_metrics = trainer.evaluate()
    print("=== Детализированные метрики ===")
    for key, value in eval_metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    train_model()
