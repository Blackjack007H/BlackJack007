from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_path = model_path = "blackjack007007/imdb-sentiment-bert" 
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return "positive" if predicted_class == 1 else "negative"

# Example 
if __name__ == "__main__":
    text = "The soundtrack was a masterpiece, perfectly complementing the emotions and atmosphere of the film"
    result = predict_sentiment(text)
    print(f"The sentiment is: {result}")
