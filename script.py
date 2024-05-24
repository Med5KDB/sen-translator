from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

sample_text = "Bonjour, comment ça va ?"
batch = tokenizer([sample_text], return_tensors="pt")
generated_ids = model.generate(**batch)
translated = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Model loaded successfully!")
print(translated)