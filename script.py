from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Helsinki-NLP/opus-mt-fr-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Model loaded successfully!")

def translateFrenchToEnglish(text):
    batch = tokenizer([text], return_tensors="pt")
    generated_ids = model.generate(**batch)
    translated = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("Traduction anglaise du texte: ", translated)
    return translated

sample_text = input("Veuillez saisir le texte que vous souhaitez modifier: ")
translateFrenchToEnglish(sample_text)
