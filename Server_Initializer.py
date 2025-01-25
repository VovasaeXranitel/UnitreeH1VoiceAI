from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === Конфигурация ===
MODEL_PATH = "/path/to/your/model"
TOKENIZER_PATH = "/path/to/your/tokenizer"

# Загрузка модели и токенизатора
print("Загрузка модели и токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.eval()  # Переводим модель в режим инференса
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

app = FastAPI()


# Модель запроса
class Query(BaseModel):
    question: str


@app.post("/predict/")
async def predict(query: Query):
    prompt = f"Вопрос: {query.question}\nОтвет:"

    inputs = tokenizer(
        prompt, return_tensors="pt", max_length=128, truncation=True, padding=True
    ).to(device)

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
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
