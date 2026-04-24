import random
import csv
import numpy as np

# ---------------------------------------------------------
# Synthetic Dataset Generator (Sentiment 0-100)
# ---------------------------------------------------------

subjects = [
    "The product", "This software", "The customer service", "My recent purchase",
    "The new update", "Their mobile app", "The user interface", "This tool",
    "The delivery process", "The premium subscription"
]

# (Word/Phrase, Sentiment Shift)
adjectives = [
    ("absolutely phenomenal", 45), ("incredibly smooth and intuitive", 40),
    ("fantastic", 35), ("very reliable", 25), ("better than expected", 20),
    ("pretty good", 10), ("okay but nothing special", 0), ("just average", 0),
    ("somewhat mediocre", -10), ("a bit disappointing", -15),
    ("clunky and slow", -25), ("full of bugs", -30), ("terrible", -35),
    ("a complete disaster", -45), ("the worst thing I have ever used", -50)
]

follow_ups = [
    ("I will recommend it to everyone I know.", 10),
    ("Best money I've spent all year.", 15),
    ("It does exactly what I need without fuss.", 5),
    ("I'm quite satisfied overall.", 5),
    ("It's fine for basic tasks.", 0),
    ("I might look for alternatives soon.", -5),
    ("Needs a lot of improvement.", -10),
    ("I am requesting a full refund.", -15),
    ("Save your money, look elsewhere.", -20)
]

def generate_sentence():
    # Base score starts in the middle
    base_score = 50.0

    sub = random.choice(subjects)
    adj, adj_score = random.choice(adjectives)
    fol, fol_score = random.choice(follow_ups)

    sentence = f"{sub} is {adj}. {fol}"

    # Calculate score and add Gaussian noise (human variance)
    final_score = base_score + adj_score + fol_score + np.random.normal(0, 3)

    # Clip strictly between 0 and 100
    final_score = max(0.0, min(100.0, final_score))

    return sentence, round(final_score, 1)

# Generate 500 samples
NUM_SAMPLES = 500
dataset = []

for _ in range(NUM_SAMPLES):
    dataset.append(generate_sentence())

# Save to CSV
filename = "company_data_500.csv"
with open(filename, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["text", "score"])
    for row in dataset:
        writer.writerow(row)

print(f"✅ Generated {NUM_SAMPLES} samples and saved to {filename}")
print(f"Sample 1: {dataset[0]}")
print(f"Sample 2: {dataset[1]}")
