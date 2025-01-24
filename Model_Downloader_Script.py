from transformers import AutoTokenizer, AutoModelForCausalLM

# === Конфигурация ===
MODEL_NAME = "ai-forever/rugpt3small_based_on_gpt2"  # Название модели
SAVE_DIR = "local_model"  # Папка для сохранения модели и токенизатора

def download_model():
    # Загрузка токенизатора
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Токенизатор сохранен в {SAVE_DIR}")

    # Загрузка модели
    print("Загрузка модели...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.save_pretrained(SAVE_DIR)
    print(f"Модель сохранена в {SAVE_DIR}")
