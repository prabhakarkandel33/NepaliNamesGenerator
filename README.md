# Nepali Name Generator (Character-Level Model)

This repository documents my journey exploring different neural network architectures to build a character-level model that generates Nepali names.

It is heavily inspired by [Andrej Karpathy's makemore](https://github.com/karpathy/makemore) project and uses similar training and sampling techniques, adapted to Nepali name datasets.

---

### 📚 Dataset

Names were sourced from:

- [techgaun/nepali-names](https://github.com/techgaun/nepali-names)

---

### 🧠 Models (in `notebooks/`)

- `itr1.ipynb`: Simple bigram model
- `itr2.ipynb`: Initial MLP
- `itr3.ipynb`: MLP with BatchNorm
- `wavenet.ipynb`: Final model using WaveNet-style convolutional layers

---

### 🔧 Final Script

- `project.py`: Cleaned version of the final training and sampling pipeline.

---

### 📄 Sample Outputs

- `sampled_*.txt`: Generated name samples from various model checkpoints.

---

This project helped me understand embeddings, context windows, overfitting, regularization (like dropout and batch norm), and how small architectural changes impact generalization in character-level models.
