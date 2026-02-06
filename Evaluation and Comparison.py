#  Evaluation and Comparison
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("=" * 70)
print("MODEL EVALUATION AND COMPARISON")
print("=" * 70)

# 1. Generate Predictions
print("\nGenerating predictions...")

# Random Forest predictions
y_pred_rf = rf_classifier.predict(X_test_tfidf)

# LSTM predictions
y_pred_lstm_probs = lstm_model.predict(X_test_seq, verbose=0)
y_pred_lstm = (y_pred_lstm_probs > 0.5).astype("int32").flatten()

# 2. Calculate Accuracy Scores
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_lstm = accuracy_score(y_test, y_pred_lstm)

print(f"\nRandom Forest Accuracy: {acc_rf:.4f} ({acc_rf*100:.2f}%)")
print(f"LSTM Accuracy: {acc_lstm:.4f} ({acc_lstm*100:.2f}%)")

# 3. Classification Reports
print("\n" + "=" * 70)
print("RANDOM FOREST CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(y_test, y_pred_rf, target_names=['Non-Actionable', 'Actionable']))

print("\n" + "=" * 70)
print("LSTM CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(y_test, y_pred_lstm, target_names=['Non-Actionable', 'Actionable']))

# 4. Confusion Matrix Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Random Forest Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Non-Actionable', 'Actionable'],
            yticklabels=['Non-Actionable', 'Actionable'],
            cbar_kws={'label': 'Count'})
axes[0].set_title(f'Random Forest Confusion Matrix\nAccuracy: {acc_rf:.2%}', 
                  fontsize=12, fontweight='bold')
axes[0].set_ylabel('Actual Label', fontsize=10)
axes[0].set_xlabel('Predicted Label', fontsize=10)

# LSTM Confusion Matrix
cm_lstm = confusion_matrix(y_test, y_pred_lstm)
sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Non-Actionable', 'Actionable'],
            yticklabels=['Non-Actionable', 'Actionable'],
            cbar_kws={'label': 'Count'})
axes[1].set_title(f'LSTM Confusion Matrix\nAccuracy: {acc_lstm:.2%}', 
                  fontsize=12, fontweight='bold')
axes[1].set_ylabel('Actual Label', fontsize=10)
axes[1].set_xlabel('Predicted Label', fontsize=10)

plt.tight_layout()
plt.show()

# 5. Performance Comparison Summary
print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON SUMMARY")
print("=" * 70)

from sklearn.metrics import precision_score, recall_score, f1_score

metrics_comparison = {
    'Model': ['Random Forest', 'LSTM'],
    'Accuracy': [acc_rf, acc_lstm],
    'Precision': [
        precision_score(y_test, y_pred_rf, average='weighted'),
        precision_score(y_test, y_pred_lstm, average='weighted')
    ],
    'Recall': [
        recall_score(y_test, y_pred_rf, average='weighted'),
        recall_score(y_test, y_pred_lstm, average='weighted')
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_rf, average='weighted'),
        f1_score(y_test, y_pred_lstm, average='weighted')
    ]
}

import pandas as pd
comparison_df = pd.DataFrame(metrics_comparison)
print("\n", comparison_df.to_string(index=False))

# Determine best model
best_model = 'Random Forest' if acc_rf > acc_lstm else 'LSTM'
print(f"\n🏆 Best Performing Model: {best_model}")
print("=" * 70)

# 6. Visual Comparison of Metrics
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(comparison_df.columns[1:]))
width = 0.35

bars1 = ax.bar(x - width/2, comparison_df.iloc[0, 1:], width, 
               label='Random Forest', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, comparison_df.iloc[1, 1:], width, 
               label='LSTM', color='#2ecc71', alpha=0.8)

ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison_df.columns[1:])
ax.legend()
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

print("\n✓ Evaluation complete!")