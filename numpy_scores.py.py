#TASK 1: Generate and Inspect the Data
import numpy as np

# Setting seed for reproducibility
np.random.seed(42)

# Generate 5x4 array with scores between 50 and 100
scores = np.random.randint(50, 101, size=(5, 4))

print("Original Scores Matrix:\n", scores)

# 1. 3rd student (index 2), 2nd subject (index 1)
val_a = scores[2, 1]

# 2. Last 2 students (index 3 onwards), all subjects
val_b = scores[-2:, :]

# 3. First 3 students (index 0:3), subjects 2 and 3 (index 1:3)
val_c = scores[0:3, 1:3]

print(f"\n3rd Student, 2nd Subject: {val_a}")
print(f"Last 2 Students:\n{val_b}")
print(f"First 3 Students (Subj 2 & 3):\n{val_c}")

#TASK 2: Analyze with Broadcasting
# 1. Column-wise mean (axis=0)
subject_means = np.round(np.mean(scores, axis=0), 2)

# 2. Broadcasting the curve [5, 3, 7, 2]
curve = np.array([5, 3, 7, 2])
curved_scores = scores + curve

# Ensure no score exceeds 100
curved_scores = np.clip(curved_scores, 0, 100)

# 3. Row-wise max (axis=1)
best_per_student = np.max(curved_scores, axis=1)

print(f"\nSubject Means: {subject_means}")
print(f"Curved Scores (Clipped at 100):\n{curved_scores}")
print(f"Best Subject Score per Student: {best_per_student}")

#TASK 3: Normalize and Identify
# 1. Min-Max Normalization per row
row_min = curved_scores.min(axis=1, keepdims=True)
row_max = curved_scores.max(axis=1, keepdims=True)

# Formula: (score - min) / (max - min)
normalized_scores = (curved_scores - row_min) / (row_max - row_min + 1e-9)

# 2. Find index of highest value
# argmax returns the flattened index, unravel_index converts it to (row, col)
highest_idx_flat = np.argmax(normalized_scores)
student_idx, subject_idx = np.unravel_index(highest_idx_flat, normalized_scores.shape)

# 3. Boolean Masking for scores > 90
scores_above_90 = curved_scores[curved_scores > 90]

print(f"\nNormalized Scores (first 2 rows):\n{normalized_scores[:2]}")
print(f"Highest Value found at: Student {student_idx}, Subject {subject_idx}")
print(f"Scores strictly above 90: {scores_above_90}")