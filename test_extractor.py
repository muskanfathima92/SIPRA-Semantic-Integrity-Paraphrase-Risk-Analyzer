from src.text_extractor import extract_text


file_path = "data/sample.txt"

text = extract_text(file_path)

print("\n========== EXTRACTED TEXT ==========\n")

print(text)

print("\n=====================================")