# Step 3: Feature Engineering (For Machine Learning)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("FEATURE ENGINEERING (FOR MACHINE LEARNING)")
print("=" * 70)

# 1. Define Features and Target
print("\n[Step 1/4] Defining Features and Target Variables")
print("-" * 70)

X = df['cleaned_text']
y = df['actionable']

print(f"✓ Features (X): cleaned_text column")
print(f"✓ Target (y): actionable column")
print(f"\nDataset Statistics:")
print(f"  - Total samples: {len(X)}")
print(f"  - Actionable samples: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
print(f"  - Non-Actionable samples: {(~y.astype(bool)).sum()} ({(~y.astype(bool)).sum()/len(y)*100:.2f}%)")

# Check for missing values
missing_X = X.isna().sum()
missing_y = y.isna().sum()

if missing_X > 0 or missing_y > 0:
    print(f"\n⚠️ Warning: Found missing values!")
    print(f"  - Missing in X: {missing_X}")
    print(f"  - Missing in y: {missing_y}")
    print(f"  Removing missing values...")
    
    # Remove missing values
    valid_indices = X.notna() & y.notna()
    X = X[valid_indices]
    y = y[valid_indices]
    
    print(f"✓ Cleaned dataset size: {len(X)}")
else:
    print(f"✓ No missing values detected")

# 2. Split Data into Train and Test Sets
print("\n[Step 2/4] Splitting Data into Train and Test Sets")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"Split Configuration:")
print(f"  - Train size: 80%")
print(f"  - Test size: 20%")
print(f"  - Random state: 42")
print(f"  - Stratified: Yes (maintains class distribution)")

print(f"\n✓ Data split completed:")
print(f"  - Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  - Testing samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

print(f"\nClass Distribution in Train Set:")
print(f"  - Actionable: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.2f}%)")
print(f"  - Non-Actionable: {(~y_train.astype(bool)).sum()} ({(~y_train.astype(bool)).sum()/len(y_train)*100:.2f}%)")

print(f"\nClass Distribution in Test Set:")
print(f"  - Actionable: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.2f}%)")
print(f"  - Non-Actionable: {(~y_test.astype(bool)).sum()} ({(~y_test.astype(bool)).sum()/len(y_test)*100:.2f}%)")

# 3. Initialize and Fit TF-IDF Vectorizer
print("\n[Step 3/4] TF-IDF Vectorization")
print("-" * 70)

# Initialize TF-IDF Vectorizer
# max_features=3000 keeps the top most frequent words to prevent overfitting
tfidf = TfidfVectorizer(
    max_features=3000, 
    ngram_range=(1, 2),
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_df=0.95,  # Ignore terms that appear in more than 95% of documents
    sublinear_tf=True  # Apply sublinear tf scaling
)

print(f"TF-IDF Configuration:")
print(f"  - Max features: 3000")
print(f"  - N-gram range: (1, 2) [unigrams + bigrams]")
print(f"  - Min document frequency: 2")
print(f"  - Max document frequency: 0.95")
print(f"  - Sublinear TF: True")

# Fit and Transform
print(f"\n✓ Fitting TF-IDF on training data...")
X_train_tfidf = tfidf.fit_transform(X_train)

print(f"✓ Transforming test data...")
X_test_tfidf = tfidf.transform(X_test)

print(f"\n✅ TF-IDF Vectorization completed!")

# 4. Display Results
print("\n[Step 4/4] Feature Matrix Statistics")
print("-" * 70)

print(f"\nTraining Features:")
print(f"  - Shape: {X_train_tfidf.shape}")
print(f"  - Samples: {X_train_tfidf.shape[0]}")
print(f"  - Features: {X_train_tfidf.shape[1]}")
print(f"  - Sparsity: {(1.0 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]))*100:.2f}%")

print(f"\nTesting Features:")
print(f"  - Shape: {X_test_tfidf.shape}")
print(f"  - Samples: {X_test_tfidf.shape[0]}")
print(f"  - Features: {X_test_tfidf.shape[1]}")
print(f"  - Sparsity: {(1.0 - X_test_tfidf.nnz / (X_test_tfidf.shape[0] * X_test_tfidf.shape[1]))*100:.2f}%")

# 5. Analyze Vocabulary
print("\n" + "=" * 70)
print("VOCABULARY ANALYSIS")
print("=" * 70)

vocabulary = tfidf.get_feature_names_out()
print(f"\n✓ Total vocabulary size: {len(vocabulary)}")

# Sample features
print(f"\nSample features (first 20):")
print(vocabulary[:20])

# Check n-grams distribution
unigrams = [word for word in vocabulary if ' ' not in word]
bigrams = [word for word in vocabulary if ' ' in word]

print(f"\nN-gram Distribution:")
print(f"  - Unigrams: {len(unigrams)} ({len(unigrams)/len(vocabulary)*100:.2f}%)")
print(f"  - Bigrams: {len(bigrams)} ({len(bigrams)/len(vocabulary)*100:.2f}%)")

print(f"\nSample bigrams (first 10):")
print(bigrams[:10] if len(bigrams) >= 10 else bigrams)

# 6. Visualize Class Distribution
print("\n" + "=" * 70)
print("DATA SPLIT VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Sample Distribution
categories = ['Training Set', 'Testing Set']
sample_counts = [len(X_train), len(X_test)]
colors = ['#3498db', '#e74c3c']

axes[0].bar(categories, sample_counts, color=colors, alpha=0.8, edgecolor='black')
axes[0].set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
axes[0].set_title('Train-Test Split Distribution', fontsize=12, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3, linestyle='--')

for i, (cat, count) in enumerate(zip(categories, sample_counts)):
    axes[0].text(i, count + 20, f'{count}\n({count/len(X)*100:.1f}%)', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Class Distribution Comparison
x_pos = np.arange(2)
width = 0.35

train_actionable = y_train.sum()
train_non_actionable = (~y_train.astype(bool)).sum()
test_actionable = y_test.sum()
test_non_actionable = (~y_test.astype(bool)).sum()

bars1 = axes[1].bar(x_pos - width/2, [train_actionable, test_actionable], 
                    width, label='Actionable', color='#2ecc71', alpha=0.8, edgecolor='black')
bars2 = axes[1].bar(x_pos + width/2, [train_non_actionable, test_non_actionable], 
                    width, label='Non-Actionable', color='#95a5a6', alpha=0.8, edgecolor='black')

axes[1].set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
axes[1].set_title('Class Distribution: Train vs Test', fontsize=12, fontweight='bold')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(['Training Set', 'Testing Set'])
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# 7. Sample TF-IDF Features
print("\n" + "=" * 70)
print("SAMPLE TF-IDF FEATURES")
print("=" * 70)

# Show TF-IDF values for first sample
sample_idx = 0
sample_text = X_train.iloc[sample_idx]
sample_vector = X_train_tfidf[sample_idx].toarray()[0]

# Get non-zero features
non_zero_indices = np.where(sample_vector > 0)[0]
non_zero_features = [(vocabulary[i], sample_vector[i]) for i in non_zero_indices]
non_zero_features.sort(key=lambda x: x[1], reverse=True)

print(f"\nSample Text: {sample_text[:150]}...")
print(f"\nTop 10 TF-IDF Features for this sample:")
print("-" * 70)

for i, (feature, score) in enumerate(non_zero_features[:10], 1):
    print(f"{i:2d}. {feature:20s} : {score:.4f}")

print("\n" + "=" * 70)
print("✅ FEATURE ENGINEERING COMPLETE!")
print("=" * 70)

# Summary
print(f"\n📊 Summary:")
print(f"  ✓ Data split into {len(X_train)} train and {len(X_test)} test samples")
print(f"  ✓ Created {X_train_tfidf.shape[1]} TF-IDF features")
print(f"  ✓ Features include unigrams and bigrams")
print(f"  ✓ Class distribution maintained through stratification")
print(f"  ✓ Ready for model training!")
print("=" * 70)