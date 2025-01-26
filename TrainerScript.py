import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === Конфигурация ===
TOKENIZER_PATH = "/home/vremennaya-kpu/.cache/huggingface/hub/models--ai-forever--rugpt3small_based_on_gpt2/rugpt3_tokenizer"
MODEL_PATH = "/home/vremennaya-kpu/.cache/huggingface/hub/models--ai-forever--rugpt3small_based_on_gpt2"  # Путь к сохраненной модели
DATASET_PATH = "/home/vremennaya-kpu/PycharmProjects/UnitreeH1VoiceAI/Вопрос-Ответ-ИИИ-Файл.json"  # Путь к данным
OUTPUT_DIR = "/home/vremennaya-kpu/.cache/huggingface/hub/Output_Fine_Tuned_Model_RUGPT_v2"  # Путь для сохранения дообученной модели

# === Функция для подготовки данных ===
def preprocess_function(examples, tokenizer):
    question = str(examples["question"]) if examples["question"] is not None else ""
    answer = str(examples["answer"]) if examples["answer"] is not None else ""

    inputs = tokenizer(
        question,
        text_pair=answer,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

# === Функция для вычисления метрик ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    labels = torch.tensor(labels)
    accuracy = accuracy_score(labels.flatten().numpy(), predictions.flatten().numpy())
    return {"accuracy": accuracy}

# === Основная функция ===
def train_model():
    # Загрузка данных
    print("Загрузка данных...")
    raw_data = load_dataset("json", data_files=DATASET_PATH)

    # Разделение данных на обучающую и валидационную выборки
    print("Разделение данных...")
    raw_train_data = raw_data["train"]
    data_list = [{"question": item["question"], "answer": item["answer"]} for item in raw_train_data]

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

    # Загрузка модели для дообучения
    print("Загрузка модели...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    # Аргументы тренировки
    print("Настройка тренировки...")
    training_args = TrainingArguments(
        learning_rate=1e-5,  # Оптимально для fine-tuning LLM
        warmup_steps=400,  # Warm-up для стабилизации
        lr_scheduler_type="cosine",  # Постепенное уменьшение LR
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        per_device_train_batch_size=10,  # Меньший размер батча для стабильности
        per_device_eval_batch_size=10,
        num_train_epochs=1000,  # Больше эпох для улучшения качества
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
        compute_metrics=compute_metrics
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