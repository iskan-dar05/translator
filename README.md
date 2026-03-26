# Spanish-French-English Translator

A **neural machine translation (NMT)** project built with **TensorFlow / Keras** to translate between English, French, and Spanish.  
Supports translations in **four directions**:  
- English → French  
- French → English  
- English → Spanish  
- Spanish → English  

This project can be trained from scratch on the Europarl dataset and also allows resuming training from saved checkpoints.

---

## Project Structure
```bash
translator/
├─ env/ # Python virtual environment
├─ nmt/ # Saved model & tokenizer
│  ├─ model.keras # Keras translation model
│  ├─ state.pkl # Training progress
│  └─ tokenizer.pkl # Tokenizer for text preprocessing
├─ dataset/ # Downloaded Europarl datasets
├─ train.py # Script to preprocess data and train model
├─ main.py # Script to run translation (inference)
├─ Spanish_Franch_English_Translator.ipynb # Jupyter notebook version
└─ README.md # Project documentation
```
---

## Features

- Multi-lingual translation (EN ↔ FR, EN ↔ ES)  
- Preprocessing with NLTK and custom cleaning  
- Supports resuming training with saved model states  
- Chunked training for large datasets  
- Custom tokenizer using Keras Tokenizer  
- Saves model in `nmt/model.keras` for inference  

---

## Requirements

- Python 3.8+  
- TensorFlow 2.x  
- Numpy  
- NLTK  
- scikit-learn  

Install dependencies and download NLTK tokenizer:

pip install tensorflow numpy nltk scikit-learn
python -c "import nltk; nltk.download('punkt')"

---

## Usage

1. Train the model:
python train.py

2. Run inference:
python main.py

Input a sentence with a language tag to translate, for example:
<en> <to_fr> Hello, how are you?
<sos> Bonjour, comment ça va ? <eos>

---

## Model Files

- `model.keras` → Keras trained model  
- `tokenizer.pkl` → Tokenizer for preprocessing input/output  
- `state.pkl` → Keeps track of training progress for resuming  

---

## Dataset

- Europarl v7 dataset: [English-French](https://www.statmt.org/europarl/v7/fr-en.tgz) | [English-Spanish](https://www.statmt.org/europarl/v7/es-en.tgz)  
- Automatically downloaded when running `train.py`  

---

## Notes

- Max sequence length: 40 tokens  
- Latent dimension: 128  
- Training chunk size: 10,000 pairs  
- Batch size: 16  
- Epochs per chunk: 1 (adjustable)  

---

## License

MIT License – feel free to use and modify.