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
    print("+---------------------------------------------------+");
    print("|                    Sen Translator                 |");
    print("+---------------------------------------------------+");
    print("|                  1. Anglais -> Francais           |");
    print("|                  2. Francais -> Anglais           |");
    print("|                  3. Quitter le programme          |");
    print("+---------------------------------------------------+");
    
    try :
        choice = input("Votre choix: ");
    except: 
        print("Erreur lors de la saisie du choix! Veuillez réessayer.");
    
    match choice:
        case "1":
            print("Vous avez choisi la traduction de l'anglais 🏴󠁧󠁢󠁥󠁮󠁧󠁿 vers le français 🇫🇷");
            
            english_to_french_model_name = "Helsinki-NLP/opus-mt-en-fr"
            english_to_french_tokenizer = AutoTokenizer.from_pretrained(english_to_french_model_name)
            english_to_french_model = AutoModelForSeq2SeqLM.from_pretrained(english_to_french_model_name)
            
            print("English to French Model loaded successfully 🎊🎉")
            
            english_text = input("Please provide the text you want to translate to french: ")
            translateEnglishToFrench(english_text)
        
        case '2':
            print("Vous avez choisi la Traduction du français 🇫🇷 vers l'anglais 🏴󠁧󠁢󠁥󠁮󠁧󠁿");
            
            french_to_english_model_name = "Helsinki-NLP/opus-mt-fr-en"
            french_to_english_tokenizer = AutoTokenizer.from_pretrained(french_to_english_model_name)
            french_to_english_model = AutoModelForSeq2SeqLM.from_pretrained(french_to_english_model_name)
            
            print("French to English Model loaded successfully 🎊🎉")
            
            french_text = input("Veuillez saisir le texte que vous souhaitez traduire en anglais: ")
            translateFrenchToEnglish(french_text)
        
        case "3":
            print("Bye 👋")
            break;
        
        case _:
            print("Veuillez saisir un choix valide!")
