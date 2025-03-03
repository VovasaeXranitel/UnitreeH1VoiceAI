from transformers import AutoTokenizer, AutoModelForCausalLM

# Название модели на Hugging Face
MODEL_NAME = "ai-forever/rugpt3small_based_on_gpt2"

def install_and_load_model():
    # Загрузка токенизатора
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Токенизатор загружен.")

    # Загрузка модели
    print("Загрузка модели...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    print("Модель загружена.")

    return tokenizer, model

if __name__ == "__main__":
    tokenizer, model = install_and_load_model()
    print("Модель и токенизатор готовы к использованию.")