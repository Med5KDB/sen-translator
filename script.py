from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def translateFrenchToEnglish(text):
    batch = french_to_english_tokenizer([text], return_tensors="pt")
    generated_ids = french_to_english_model.generate(**batch)
    translated = french_to_english_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("Traduction anglaise du texte: ", translated)
    return translated

def translateEnglishToFrench(text):
    batch = english_to_french_tokenizer([text], return_tensors="pt")
    generated_ids = english_to_french_model.generate(**batch)
    translated = english_to_french_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("Traduction française du texte: ", translated)
    return translated

while True:
    choice = input("Voulez-vous traduire du français vers l'anglais ou de l'anglais vers le français? (fr/en): ")
    if choice == "fr":
        french_to_english_model_name = "Helsinki-NLP/opus-mt-fr-en"
        french_to_english_tokenizer = AutoTokenizer.from_pretrained(french_to_english_model_name)
        french_to_english_model = AutoModelForSeq2SeqLM.from_pretrained(french_to_english_model_name)
        print("French to English Model loaded successfully!")
        
        french_text = input("Veuillez saisir le texte que vous souhaitez traduire en anglais: ")
        translateFrenchToEnglish(french_text)
    elif choice == "en":
        english_to_french_model_name = "Helsinki-NLP/opus-mt-en-fr"
        english_to_french_tokenizer = AutoTokenizer.from_pretrained(english_to_french_model_name)
        english_to_french_model = AutoModelForSeq2SeqLM.from_pretrained(english_to_french_model_name)
        print("English to French Model loaded successfully!")
        
        english_text = input("Please provide the text you want to translate to french: ")
        translateEnglishToFrench(english_text)
    else:
        print("Choix invalide!")
