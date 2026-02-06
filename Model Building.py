# Step 4: Machine Learning Model Building (Random Forest)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

print("=" * 70)
print("MACHINE LEARNING MODEL BUILDING (RANDOM FOREST)")
print("=" * 70)

# 1. Initialize Random Forest Classifier
print("\n[Step 1/3] Initializing Random Forest Classifier")
print("-" * 70)

rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    verbose=0
)

print("✓ Random Forest Classifier initialized")
print(f"  - Number of trees: 100")
print(f"  - Random state: 42")
print(f"  - Parallel jobs: All available cores")

# 2. Train the Model
print("\n[Step 2/3] Training Random Forest Classifier")
print("-" * 70)

print(f"Training data shape: {X_train_tfidf.shape}")
print(f"Training labels shape: {y_train.shape}")
print("\nTraining in progress...")

rf_classifier.fit(X_train_tfidf, y_train)

print("\n✅ Training completed successfully!")

# 3. Quick Performance Check
print("\n[Step 3/3] Model Performance Summary")
print("-" * 70)

# Training accuracy
train_pred = rf_classifier.predict(X_train_tfidf)
train_accuracy = accuracy_score(y_train, train_pred)

# Test accuracy
test_pred = rf_classifier.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, test_pred)

print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Check for overfitting
accuracy_gap = train_accuracy - test_accuracy
if accuracy_gap > 0.1:
    print(f"\n⚠️ Warning: Potential overfitting detected (gap: {accuracy_gap:.4f})")
else:
    print(f"\n✓ Model generalization looks good (gap: {accuracy_gap:.4f})")

# 4. Feature Importance Analysis
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

# Get feature importances
importances = rf_classifier.feature_importances_
feature_names = tfidf.get_feature_names_out()

print(f"\n✓ Total features analyzed: {len(feature_names)}")
print(f"✓ Non-zero importance features: {np.sum(importances > 0)}")

# Get the top 15 features
top_n = 15
indices = np.argsort(importances)[-top_n:]

print(f"\nTop {top_n} Most Important Features:")
print("-" * 70)

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({
    'Feature': [feature_names[i] for i in indices][::-1],
    'Importance': importances[indices][::-1]
})

print(feature_importance_df.to_string(index=False))

# 5. Visualize Feature Importance
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Horizontal Bar Chart (Top 15)
axes[0].barh(range(len(indices)), importances[indices], color='teal', alpha=0.8, edgecolor='black')
axes[0].set_yticks(range(len(indices)))
axes[0].set_yticklabels([feature_names[i] for i in indices], fontsize=10)
axes[0].set_xlabel('Importance Score', fontsize=11, fontweight='bold')
axes[0].set_title(f'Top {top_n} Most Important Features for Actionable Detection', 
                  fontsize=12, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, v in enumerate(importances[indices]):
    axes[0].text(v + 0.0001, i, f'{v:.4f}', va='center', fontsize=8)

# Plot 2: Vertical Bar Chart with all top features
top_20_indices = np.argsort(importances)[-20:]
axes[1].bar(range(len(top_20_indices)), importances[top_20_indices], 
            color='steelblue', alpha=0.8, edgecolor='black')
axes[1].set_xticks(range(len(top_20_indices)))
axes[1].set_xticklabels([feature_names[i] for i in top_20_indices], 
                        rotation=45, ha='right', fontsize=9)
axes[1].set_ylabel('Importance Score', fontsize=11, fontweight='bold')
axes[1].set_title('Top 20 Features - Importance Distribution', 
                  fontsize=12, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

# 6. Additional Feature Statistics
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE STATISTICS")
print("=" * 70)

print(f"\nImportance Statistics:")
print(f"  - Maximum importance: {importances.max():.6f}")
print(f"  - Minimum importance: {importances.min():.6f}")
print(f"  - Mean importance: {importances.mean():.6f}")
print(f"  - Median importance: {np.median(importances):.6f}")
print(f"  - Standard deviation: {importances.std():.6f}")

# Cumulative importance
sorted_importances = np.sort(importances)[::-1]
cumulative_importance = np.cumsum(sorted_importances)

# Find how many features contribute to 80% of importance
features_for_80_percent = np.argmax(cumulative_importance >= 0.8) + 1
print(f"\n✓ Top {features_for_80_percent} features contribute to 80% of total importance")

# 7. Cumulative Importance Plot
print("\n" + "=" * 70)
print("CUMULATIVE FEATURE IMPORTANCE")
print("=" * 70)

plt.figure(figsize=(12, 6))
plt.plot(range(1, len(sorted_importances) + 1), cumulative_importance, 
         linewidth=2, color='darkgreen', marker='o', markersize=3, markevery=50)
plt.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='80% Threshold')
plt.axvline(x=features_for_80_percent, color='orange', linestyle='--', 
            linewidth=2, label=f'{features_for_80_percent} Features')
plt.xlabel('Number of Features', fontsize=11, fontweight='bold')
plt.ylabel('Cumulative Importance', fontsize=11, fontweight='bold')
plt.title('Cumulative Feature Importance', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

# 8. Sample Predictions
print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

# Show a few predictions
sample_size = 5
sample_indices = np.random.choice(len(X_test), sample_size, replace=False)

print(f"\nShowing {sample_size} random predictions from test set:\n")

for idx, sample_idx in enumerate(sample_indices, 1):
    text = X_test.iloc[sample_idx]
    actual = y_test.iloc[sample_idx]
    predicted = test_pred[sample_idx]
    
    print(f"{idx}. Text: {text[:100]}...")
    print(f"   Actual: {'Actionable' if actual == 1 else 'Non-Actionable'}")
    print(f"   Predicted: {'Actionable' if predicted == 1 else 'Non-Actionable'}")
    print(f"   {'✓ Correct' if actual == predicted else '✗ Incorrect'}")
    print()

print("=" * 70)
print("✅ RANDOM FOREST MODEL BUILDING COMPLETE!")
print("=" * 70)