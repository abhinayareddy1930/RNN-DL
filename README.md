# Latin-to-Devanagari Transliteration using Seq2Seq (LSTM)

This project trains a sequence-to-sequence (seq2seq) LSTM-based deep learning model for transliterating text written in **Latin script** into **Devanagari script** (used for Hindi and related languages).

It uses the [Dakshina dataset](https://github.com/google-research-datasets/dakshina) by Google Research.

##  Model Architecture

- **Encoder**: LSTM network that encodes Latin characters into a context vector.
- **Decoder**: LSTM network that generates Devanagari script characters based on the encoder’s context.
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam / RMSprop
- **Output**: Devanagari script string corresponding to a given Latin input.

---

##  Dataset

The dataset used is from the Dakshina project. It contains transliteration pairs (native script and Latin script) for many South Asian languages.

### Example (hi.translit.sampled.train.tsv):

- Columns:
  1. `native`: Devanagari text
  2. `latin`: Latin transliteration
  3. `attested`: Whether the mapping is attested or synthetic

---

---

##  Requirements

- Python 3.x
- TensorFlow
- NumPy
- pandas
- requests (for downloading dataset)
- Google Colab (optional for training)

Install required packages:

```bash
pip install tensorflow numpy pandas requests

📊 Accuracy
Test Accuracy: 88.45%

✅ Sample Predictions

Latin	Predicted
priyanka	प्रियांका
bharat	भारत
kumar	कुमार
delhi	दिल्ली
ashok	अशोक
```
#  GPT-2 Fine-Tuning for Song Lyric Generation

This notebook fine-tunes OpenAI's GPT-2 model to generate song lyrics. It uses the Hugging Face `transformers` and `datasets` libraries for both training and inference.

---

##  Overview

- Fine-tunes GPT-2 (`gpt2`) on a user-uploaded `lyrics.txt` file.
- Preprocesses data: removes blank lines, tokenizes to a fixed length.
- Uses Hugging Face `Trainer` API with **Causal Language Modeling**.
- Saves the model and uses the `pipeline` API for text generation.

---

##  Requirements

Install dependencies:

```bash
pip install transformers datasets
```
 ## Model & Training Settings
Model: GPT-2 (gpt2)

Tokenizer: GPT2Tokenizer (padding token set to eos_token)

Token Length: 128

Batch Size: 4

Epochs: 3

Loss Function: Causal Language Modeling (CLM)

Trainer API: Hugging Face Trainer



##  How It Works
Upload your lyrics.txt file (one lyric per line, no blank lines).

Text is cleaned and converted to a Hugging Face Dataset.

GPT-2 is fine-tuned using the Trainer API.

After training:

The model and tokenizer are saved.

The pipeline API is initialized for generation.

Enter a text prompt to generate new lyrics!

##  Sample Usage
Prompt:
dancing in the moonlight
Generated Output:
dancing in the moonlight, feeling like I'm free  
stars above are singing just for me...


