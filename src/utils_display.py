import matplotlib.pyplot as plt
from PIL import Image
import random

def show_random_subject_emotions(df):
    # Pick a random subject
    subject = random.choice(df["subject"].unique())

    # Get all images for that subject
    subject_df = df[df["subject"] == subject].sort_values("emotion")

    print(f"Subject: {subject}")

    fig, axes = plt.subplots(1, len(subject_df), figsize=(18, 3))

    if len(subject_df) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, subject_df.iterrows()):
        img = Image.open(row["path"])

        ax.imshow(img)
        ax.set_title(row["emotion"])
        ax.axis("off")

    plt.tight_layout()
    plt.show()