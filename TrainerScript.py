import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import DataCollatorForLanguageModeling

# === Конфигурация ===
MODEL_PATH = r"C:\Users\Vovas\.cache\huggingface\hub\Output_Fine_Tuned_Model_RUGPT"  # Путь к сохраненной модели (база для дообучения)
DATASET_PATH = r"C:\UnitreeH1VoiceAI\Вопрос-Ответ-ИИИ-Файл.json"                      # Путь к данным
OUTPUT_DIR = r"C:\Users\Vovas\.cache\huggingface\hub\Output_Fine_Tuned_Model_RUGPT_v2"  # Путь для сохранения дообученной модели

# === Функция для подготовки данных (объединяем вопрос и ответ в один prompt) ===
def preprocess_function(examples, tokenizer):
    question = str(examples["question"]) if examples["question"] else ""
    answer   = str(examples["answer"])   if examples["answer"] else ""

    # Формируем единый текст: "Вопрос: ...\nОтвет: ..."
    prompt_text = f"Вопрос: {question}\nОтвет: {answer}"

    # Токенизация
    inputs = tokenizer(
        prompt_text,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    # Для GPT-голов (causal LM) labels = input_ids
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

# === Функция для вычисления метрик (условная accuracy, не лучший вариант для LM) ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    labels = torch.tensor(labels)

    accuracy = accuracy_score(labels.flatten().numpy(), predictions.flatten().numpy())
    return {"accuracy": accuracy}

# === Основная функция обучения ===
def train_model():
    # 1. Загрузка датасета
    print("Загрузка данных...")
    raw_data = load_dataset("json", data_files=DATASET_PATH)

    # 2. Разделение на train/val
    print("Разделение данных...")
    raw_train_data = raw_data["train"]
    data_list = [
        {"question": item["question"], "answer": item["answer"]}
        for item in raw_train_data
    ]
    train_data_list, val_data_list = train_test_split(data_list, test_size=0.1, random_state=42)
    train_data = Dataset.from_list(train_data_list)
    val_data   = Dataset.from_list(val_data_list)

    print("Пример данных перед токенизацией:")
    print(train_data[:5])

    # 3. Загрузка токенизатора
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # 4. Токенизация данных
    print("Токенизация данных...")
    tokenized_train_data = train_data.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=["question", "answer"]
    )
    tokenized_val_data = val_data.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=["question", "answer"]
    )

    # 5. Загрузка модели
    print("Загрузка модели...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    # 6. Создаём дата-коллатор для LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-моделям не нужен MLM, а обычный causal LM
    )

    # 7. Настройки обучения (TrainingArguments)
    print("Настройка тренировки...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,  # Можно варьировать, 3-5 обычно достаточно
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=10,
        report_to="none"  # чтобы не требовать wandb и прочее
    )

    # 8. Инициализация Trainer
    print("Инициализация Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_val_data,
        tokenizer=tokenizer,
        data_collator=data_collator,      # используем дата-коллатор
        compute_metrics=compute_metrics   # оставим условную метрику accuracy
    )

    # 9. Обучение
    print("Начало обучения...")
    trainer.train()

    # 10. Сохранение модели
    print("Сохранение модели...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 11. Оценка модели (evaluation)
    print("Оценка модели...")
    eval_metrics = trainer.evaluate()
    print("=== Детализированные метрики ===")
    for key, value in eval_metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    train_model()
