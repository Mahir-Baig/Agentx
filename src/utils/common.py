
import re

def clean_extracted_text(text):
    """
    Cleans and restructures text by:
    - Removing empty lines
    - Merging lines that are part of the same sentence
    - Ensuring proper sentence structure
    
    Parameters:
    text (str): The raw extracted text from a PDF or other file format
    
    Returns:
    str: Cleaned and restructured text
    """
    # Split the text into lines
    lines = text.splitlines()

    # Remove empty lines and strip extra spaces
    lines = [line.strip() for line in lines if line.strip()]

    # Initialize variables for final text and the current sentence
    cleaned_text = []
    current_sentence = ""

    # Regular expression for checking if a line ends with a sentence-ending punctuation
    sentence_end = re.compile(r'[.!?]$')

    for line in lines:
        if sentence_end.search(line):
            # If the line ends with punctuation, consider it the end of a sentence
            current_sentence += line
            cleaned_text.append(current_sentence.strip())
            current_sentence = ""
        else:
            # Otherwise, treat it as part of the same sentence
            current_sentence += line + " "

    # If there's leftover text, add it to the cleaned text
    if current_sentence:
        cleaned_text.append(current_sentence.strip())

    # Join all the cleaned sentences into a single string
    return "\n".join(cleaned_text)