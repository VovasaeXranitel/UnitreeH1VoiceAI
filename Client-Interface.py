import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = r"C:\Users\Vovas\.cache\huggingface\hub\Output_Fine_Tuned_Model_RUGPT_v2"


def chat_with_model():
    print("Загрузка модели и токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    # Если у вас нет отдельного <PAD> токена, приравниваем паддинг к EOS
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Модель загружена. Введите 'exit' для выхода.")

    while True:
        user_input = input("Вы: ")
        if user_input.lower() == "exit":
            print("Завершение общения.")
            break

        # Формируем промпт (как обучали):
        prompt = f"Вопрос: {user_input}\nОтвет:"

        # Токенизация
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True  # чтобы был attention_mask
        ).to(device)

        # Генерация ответа
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=256,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.8,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )

        # Декодируем
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # При желании можно убрать фразы "Вопрос:" / "Ответ:", но аккуратно
        # response = response.replace("Вопрос:", "").replace("Ответ:", "")

        print(f"Нейронка: {response}\n")


if __name__ == "__main__":
    chat_with_model()
