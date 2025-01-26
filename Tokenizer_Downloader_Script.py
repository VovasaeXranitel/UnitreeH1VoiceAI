from transformers import AutoTokenizer

# === Конфигурация ===
MODEL_NAME = "ai-forever/rugpt3small_based_on_gpt2"  # Имя модели на Hugging Face
TOKENIZER_SAVE_PATH = "/home/vremennaya-kpu/.cache/huggingface/hub/rugpt3_tokenizer"  # Путь для сохранения токенизатора

def setup_tokenizer():
    # Загрузка токенизатора
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Сохранение токенизатора локально
    print("Сохранение токенизатора...")
    tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)

    print(f"Токенизатор сохранён в {TOKENIZER_SAVE_PATH}")
    return tokenizer

if __name__ == "__main__":
    tokenizer = setup_tokenizer()

    # Пример использования токенизатора
    text = "Привет, как у тебя дела?"
    tokens = tokenizer(text, return_tensors="pt")
    print("Токены:", tokens)