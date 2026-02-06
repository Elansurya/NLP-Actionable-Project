# Deep Learning Model Building (LSTM)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

print("=" * 70)
print("DEEP LEARNING MODEL BUILDING (LSTM)")
print("=" * 70)

# 1. Tokenization and Padding
print("\n[Step 1/4] Tokenization and Sequence Preparation")
print("-" * 70)

max_words = 10000
max_sequence_len = 50

print(f"Configuration:")
print(f"  - Max vocabulary size: {max_words}")
print(f"  - Max sequence length: {max_sequence_len}")

# Initialize and fit tokenizer
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

print(f"\n✓ Tokenizer fitted on training data")
print(f"  - Total unique words found: {len(tokenizer.word_index)}")
print(f"  - Vocabulary size (after limit): {min(max_words, len(tokenizer.word_index))}")

# Convert texts to sequences
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

print(f"\n✓ Texts converted to sequences")
print(f"  - Training sequences: {len(X_train_sequences)}")
print(f"  - Test sequences: {len(X_test_sequences)}")

# Pad sequences to uniform length
X_train_seq = pad_sequences(X_train_sequences, maxlen=max_sequence_len, padding='post', truncating='post')
X_test_seq = pad_sequences(X_test_sequences, maxlen=max_sequence_len, padding='post', truncating='post')

print(f"\n✓ Sequences padded to uniform length")
print(f"  - X_train_seq shape: {X_train_seq.shape}")
print(f"  - X_test_seq shape: {X_test_seq.shape}")

# Sample visualization
print(f"\nSample sequence (first 10 tokens):")
print(f"  Original text: {X_train.iloc[0][:100]}...")
print(f"  Tokenized: {X_train_seq[0][:10]}")

# 2. Build LSTM Architecture
print("\n" + "=" * 70)
print("[Step 2/4] Building LSTM Architecture")
print("-" * 70)

lstm_model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_sequence_len, name='embedding'),
    SpatialDropout1D(0.2, name='spatial_dropout'),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2, name='lstm'),
    Dense(32, activation='relu', name='dense_hidden'),
    Dropout(0.3, name='dropout'),
    Dense(1, activation='sigmoid', name='output')  # Binary output
])

# Compile the model
lstm_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("\n✓ Model architecture built successfully!")
print("\nModel Summary:")
lstm_model.summary()

# Count parameters
trainable_params = np.sum([np.prod(v.get_shape()) for v in lstm_model.trainable_weights])
print(f"\nTotal trainable parameters: {trainable_params:,}")

# 3. Setup Callbacks
print("\n" + "=" * 70)
print("[Step 3/4] Setting up Training Callbacks")
print("-" * 70)

# Early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1,
    mode='min'
)

# Reduce learning rate when stuck
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=0.00001,
    verbose=1,
    mode='min'
)

# Save best model
import os
if not os.path.exists('../models'):
    os.makedirs('../models')

checkpoint = ModelCheckpoint(
    '../models/best_lstm_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

callbacks_list = [early_stop, reduce_lr, checkpoint]

print("✓ Callbacks configured:")
print("  - EarlyStopping (patience=3)")
print("  - ReduceLROnPlateau (factor=0.5, patience=2)")
print("  - ModelCheckpoint (saves best model)")

# 4. Training
print("\n" + "=" * 70)
print("[Step 4/4] Training LSTM Model")
print("=" * 70)

print(f"\nTraining configuration:")
print(f"  - Epochs: 5")
print(f"  - Batch size: 64")
print(f"  - Validation split: 10%")
print(f"  - Training samples: {len(X_train_seq)}")
print(f"  - Validation samples: {int(len(X_train_seq) * 0.1)}")

print("\nStarting training...\n")

history = lstm_model.fit(
    X_train_seq, 
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    callbacks=callbacks_list,
    verbose=1
)

print("\n✅ Training completed!")

# 5. Training History Visualization
print("\n" + "=" * 70)
print("TRAINING HISTORY")
print("=" * 70)

# Display final metrics
final_train_loss = history.history['loss'][-1]
final_train_acc = history.history['accuracy'][-1]
final_val_loss = history.history['val_loss'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print(f"\nFinal Training Metrics:")
print(f"  - Loss: {final_train_loss:.4f}")
print(f"  - Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")

print(f"\nFinal Validation Metrics:")
print(f"  - Loss: {final_val_loss:.4f}")
print(f"  - Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot accuracy
axes[0].plot(history.history['accuracy'], label='Training Accuracy', marker='o', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s', linewidth=2)
axes[0].set_title('Model Accuracy Over Epochs', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=10)
axes[0].set_ylabel('Accuracy', fontsize=10)
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

# Plot loss
axes[1].plot(history.history['loss'], label='Training Loss', marker='o', linewidth=2, color='coral')
axes[1].plot(history.history['val_loss'], label='Validation Loss', marker='s', linewidth=2, color='red')
axes[1].set_title('Model Loss Over Epochs', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=10)
axes[1].set_ylabel('Loss', fontsize=10)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. Evaluate on Test Set
print("\n" + "=" * 70)
print("TEST SET EVALUATION")
print("=" * 70)

test_loss, test_accuracy = lstm_model.evaluate(X_test_seq, y_test, verbose=0)

print(f"\nTest Set Performance:")
print(f"  - Loss: {test_loss:.4f}")
print(f"  - Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Check for overfitting
accuracy_diff = final_train_acc - test_accuracy
if accuracy_diff > 0.1:
    print(f"\n⚠️ Warning: Potential overfitting detected!")
    print(f"  Training-Test accuracy gap: {accuracy_diff:.4f}")
else:
    print(f"\n✓ Model generalization looks good!")
    print(f"  Training-Test accuracy gap: {accuracy_diff:.4f}")

print("\n" + "=" * 70)
print("✅ LSTM MODEL BUILDING COMPLETE!")
print("=" * 70)