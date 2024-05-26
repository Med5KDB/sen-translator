from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pyfiglet
from colorama import Fore, Back, Style
import random


def translateFrenchToEnglish(text):
    batch = fr2en_tokenizer([text], return_tensors="pt")
    generated_ids = fr2en_model.generate(**batch)
    translated = fr2en_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("Traduction anglaise du texte: ", translated)
    return translated

def translateEnglishToFrench(text):
    batch = en2fr_tokenizer([text], return_tensors="pt")
    generated_ids = en2fr_model.generate(**batch)
    translated = en2fr_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("Traduction française du texte: ", translated)
    return translated

ascii_banner = pyfiglet.figlet_format("Sen Translator")
coloredText = Fore.MAGENTA + ascii_banner
print(coloredText)

while True:

    print("+---------------------------------------------------+")
    print("|          Sen Translator                           |")
    print("+---------------------------------------------------+")
    print("|         1. Anglais -> Francais                    |")
    print("|         2. Francais -> Anglais                    |")
    print("|         3. Quitter le programme                   |")
    print("+---------------------------------------------------+")

    try :
        choice = input("Votre choix: ");
    except:
        print("Erreur lors de la saisie du choix! Veuillez réessayer.");

    match choice:
        case "1":
            print("Vous avez choisi la traduction de l'anglais 🏴󠁧󠁢󠁥󠁮󠁧󠁿 vers le français 🇫🇷");

            en2fr_model_name = "Helsinki-NLP/opus-mt-en-fr"
            en2fr_tokenizer = AutoTokenizer.from_pretrained(en2fr_model_name)
            en2fr_model = AutoModelForSeq2SeqLM.from_pretrained(en2fr_model_name)

            print("English to French Model loaded successfully 🎊🎉")

            english_text = input("Please provide the text you want to translate to french: ")
            translateEnglishToFrench(english_text)

        case '2':
            print("Vous avez choisi la Traduction du français 🇫🇷 vers l'anglais 🏴󠁧󠁢󠁥󠁮󠁧󠁿");

            fr2en_model_name = "Helsinki-NLP/opus-mt-fr-en"
            fr2en_tokenizer = AutoTokenizer.from_pretrained(fr2en_model_name)
            fr2en_model = AutoModelForSeq2SeqLM.from_pretrained(fr2en_model_name)

            print("French to English Model loaded successfully 🎊🎉")

            french_text = input("Veuillez saisir le texte que vous souhaitez traduire en anglais: ")
            translateFrenchToEnglish(french_text)

        case "3":
            print("Bye 👋")
            break;

        case _:
            print("Veuillez saisir un choix valide!")
