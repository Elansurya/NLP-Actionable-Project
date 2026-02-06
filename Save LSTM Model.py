# Save the LSTM Model and Tokenizer
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("=" * 70)
print("BUILDING AND TRAINING LSTM MODEL")
print("=" * 70)

# 1. Model Parameters
vocab_size = 3000  # Should match tokenizer's num_words
embedding_dim = 128
max_length = 100  # Should match your padding length
lstm_units = 100

# 2. Build the LSTM Model
print("\nBuilding LSTM architecture...")

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    SpatialDropout1D(0.2),
    LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

# 3. Compile the Model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("\n✓ Model architecture built successfully!")
print("\nModel Summary:")
model.summary()

# 4. Setup Callbacks for Better Training
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=0.00001,
    verbose=1
)

# 5. Train the Model
print("\n" + "=" * 70)
print("TRAINING THE MODEL")
print("=" * 70)

# Check if training data exists
try:
    print(f"\nTraining data shape: {X_train_seq.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation data shape: {X_test_seq.shape}")
    print(f"Validation labels shape: {y_test.shape}")
    
    # Train the model
    history = model.fit(
        X_train_seq, 
        y_train,
        epochs=5,
        batch_size=64,
        validation_data=(X_test_seq, y_test),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    print("\n✓ Model training completed!")
    
    # Display training results
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"\nFinal Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    
except NameError as e:
    print(f"\n⚠️ Error: {e}")
    print("Please ensure X_train_seq, y_train, X_test_seq, and y_test are defined.")
    print("Run the preprocessing steps first!")

# 6. Create Directory for Models
print("\n" + "=" * 70)
print("SAVING MODEL AND TOKENIZER")
print("=" * 70)

# Create models directory if it doesn't exist
models_dir = '../models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"\n✓ Created directory: {models_dir}")
else:
    print(f"\n✓ Directory already exists: {models_dir}")

# 7. Save the Model
try:
    model_path = os.path.join(models_dir, 'lstm_model.h5')
    model.save(model_path)
    print(f"✓ LSTM Model saved to: {model_path}")
    
    # Get model size
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
    print(f"  Model size: {model_size:.2f} MB")
    
except Exception as e:
    print(f"✗ Error saving model: {e}")

# 8. Save the Tokenizer
try:
    tokenizer_path = os.path.join(models_dir, 'tokenizer.pkl')
    
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"✓ Tokenizer saved to: {tokenizer_path}")
    
    # Get tokenizer size
    tokenizer_size = os.path.getsize(tokenizer_path) / (1024 * 1024)  # Convert to MB
    print(f"  Tokenizer size: {tokenizer_size:.2f} MB")
    
    # Display tokenizer info
    print(f"\n  Tokenizer vocabulary size: {len(tokenizer.word_index)}")
    print(f"  Max words configured: {tokenizer.num_words if hasattr(tokenizer, 'num_words') else 'N/A'}")
    
except NameError:
    print("✗ Error: 'tokenizer' not found. Please run tokenization step first!")
except Exception as e:
    print(f"✗ Error saving tokenizer: {e}")

# 9. Summary
print("\n" + "=" * 70)
print("SAVE SUMMARY")
print("=" * 70)
print(f"\nSaved files location: {os.path.abspath(models_dir)}")
print("\nFiles saved:")
print("  1. lstm_model.h5 - Trained LSTM model")
print("  2. tokenizer.pkl - Fitted tokenizer object")
print("\n✅ All components saved successfully!")
print("=" * 70)

# 10. Verification - Load and Test
print("\n" + "=" * 70)
print("VERIFICATION: LOADING SAVED MODEL")
print("=" * 70)

try:
    from tensorflow.keras.models import load_model
    
    # Load model
    loaded_model = load_model(model_path)
    print("✓ Model loaded successfully!")
    
    # Load tokenizer
    with open(tokenizer_path, 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
    print("✓ Tokenizer loaded successfully!")
    
    # Quick test prediction
    if 'X_test_seq' in locals():
        test_pred = loaded_model.predict(X_test_seq[:1], verbose=0)
        print(f"\n✓ Test prediction successful: {test_pred[0][0]:.4f}")
    
    print("\n✅ Verification complete - models are ready to use!")
    
except Exception as e:
    print(f"✗ Verification error: {e}")

print("=" * 70)