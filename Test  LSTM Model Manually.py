#  Test the LSTM Model Manually
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def predict_lstm(text, model, tokenizer, clean_text, maxlen=100):
    """
    Predict whether text is Actionable or Non-Actionable using LSTM model
    
    Parameters:
    - text: Input text to classify
    - model: Trained LSTM model
    - tokenizer: Fitted tokenizer object
    - clean_text: Text cleaning function
    - maxlen: Maximum sequence length (should match training)
    
    Returns:
    - result: Classification result
    - pred_prob: Prediction probability
    """
    # 1. Clean and Tokenize
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    
    # 2. Pad sequences
    padded = pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
    
    # 3. Predict
    pred_prob = model.predict(padded, verbose=0)[0][0]
    result = "Actionable" if pred_prob > 0.5 else "Non-Actionable"
    
    # Display results
    print(f"Input: {text}")
    print(f"Cleaned: {cleaned}")
    print(f"Deep Learning Prediction: {result} ({pred_prob:.2%})")
    print("-" * 60)
    
    return result, pred_prob

# Test with multiple customer complaints
test_cases = [
    "I need help with my login, it is not working.",
    "Thank you for your excellent service!",
    "My order hasn't arrived yet and it's been 2 weeks",
    "The product quality is amazing",
    "I want a refund immediately, this is unacceptable",
    "Great experience shopping here"
]

print("=" * 60)
print("LSTM MODEL PREDICTIONS")
print("=" * 60)

for test_text in test_cases:
    predict_lstm(test_text, model, tokenizer, clean_text, maxlen=100)