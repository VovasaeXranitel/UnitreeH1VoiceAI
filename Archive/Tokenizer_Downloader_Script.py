from transformers import AutoTokenizer

# === Конфигурация ===
MODEL_NAME = "ai-forever/rugpt3small_based_on_gpt2"  # Имя модели на Hugging Face
TOKENIZER_SAVE_PATH = "./rugpt3_tokenizer"  # Путь для сохранения токенизатора

def load_and_save_tokenizer():
    """
    Загружает токенизатор для модели и сохраняет его локально.
    """
    try:
        print("Загрузка токенизатора...")
        # Загружаем токенизатор с Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        print("Сохранение токенизатора локально...")
        # Сохраняем токенизатор в указанную директорию
        tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)

        print(f"Токенизатор успешно сохранён в {TOKENIZER_SAVE_PATH}")
        return tokenizer
    except Exception as e:
        print(f"Ошибка при загрузке или сохранении токенизатора: {e}")
        return None

if __name__ == "__main__":
    tokenizer = load_and_save_tokenizer()

    # Тестирование токенизатора
    if tokenizer:
        test_text = "Привет, как дела у модели?"
        print(f"Тестовый текст: {test_text}")

        # Токенизация текста
        encoded = tokenizer(test_text, return_tensors="pt")
        print("Токенизированный результат:")
        print(encoded)