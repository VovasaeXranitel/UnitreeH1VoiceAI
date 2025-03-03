import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from accelerate import Accelerator

# === Конфигурация ===
CSV_FILE = r"C:\Users\Vovas\Downloads\NPlus1\newmetadata.csv"  # Путь к CSV файлу
TEXTS_DIR = r"C:\Users\Vovas\Downloads\NPlus1\texts"  # Папка с текстами
TAGGED_TEXTS_DIR = r"C:\Users\Vovas\Downloads\NPlus1\tagged_texts"  # Папка с аннотированными текстами
MODEL_PATH = r"fine_tuned_model_output"  # Путь к сохраненной модели
OUTPUT_DIR = r"fine_tuned_model_output_v2"  # Путь для сохранения дообученной модели


# === Функция для загрузки данных ===
def load_data(csv_file, texts_dir, tagged_texts_dir):
    # Чтение CSV файла
    metadata = pd.read_csv(csv_file)

    # Список для хранения данных
    data = []

    # Чтение текстов из папок
    for _, row in metadata.iterrows():
        text_id = row["textid"]

        # Составление путей к файлам
        text_file = os.path.join(texts_dir, f"{text_id}.txt")
        tagged_text_file = os.path.join(tagged_texts_dir, f"{text_id}.txt")

        # Чтение текстов
        text = open(text_file, encoding="utf-8").read() if os.path.exists(text_file) else None
        tagged_text = open(tagged_text_file, encoding="utf-8").read() if os.path.exists(tagged_text_file) else None

        # Добавление в список
        if text:
            data.append({"text": text, "tagged_text": tagged_text})

    # Преобразование в pandas DataFrame
    return pd.DataFrame(data)


# === Функция для подготовки данных ===
def preprocess_function(examples, tokenizer):
    inputs = tokenizer(
        examples["text"],
        text_pair=examples["tagged_text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs


# === Основная функция ===
def train_model():
    # Загрузка данных
    print("Загрузка данных...")
    raw_data = load_data(CSV_FILE, TEXTS_DIR, TAGGED_TEXTS_DIR)

    # Разделение данных на обучающую и валидационную выборки
    print("Разделение данных...")
    train_data, val_data = train_test_split(raw_data, test_size=0.1, random_state=42)

    # Подготовка токенизатора
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Токенизация данных
    print("Токенизация данных...")
    tokenized_train_data = Dataset.from_pandas(train_data).map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True
    )
    tokenized_val_data = Dataset.from_pandas(val_data).map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True
    )

    # Загрузка модели для дообучения
    print("Загрузка модели...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    # Конфигурация Accelerate
    accelerator = Accelerator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    train_dataloader = torch.utils.data.DataLoader(tokenized_train_data, batch_size=8, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(tokenized_val_data, batch_size=8)
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        val_dataloader
    )

    # Аргументы тренировки
    print("Настройка тренировки...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=10,
        report_to="none"
    )

    # Инициализация Trainer
    print("Инициализация Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_val_data,
        tokenizer=tokenizer,
    )

    # Обучение модели
    print("Начало обучения...")
    trainer.train()

    # Сохранение модели
    print("Сохранение модели...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Вывод метрик
    print("Оценка модели...")
    eval_metrics = trainer.evaluate()
    print("=== Детализированные метрики ===")
    for key, value in eval_metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    train_model()
