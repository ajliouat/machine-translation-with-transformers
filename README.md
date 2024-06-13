# Machine Translation with Transformers

This repository contains a highly sophisticated implementation of a French to English machine translation model using the Transformer architecture. The code utilizes a public dataset, GPU optimization, and incorporates advanced techniques for improved performance.

## Features

- Implements the Transformer architecture with multi-head attention, positional encoding, and separate encoder and decoder layers.
- Utilizes the BERT tokenizer for efficient text preprocessing and encoding.
- Includes a custom dataset class for handling parallel French and English sentences.
- Generates masks for source and target sequences to handle variable-length inputs and prevent future information leakage in the decoder.
- Utilizes the AdamW optimizer with a linear learning rate scheduler and warmup for training.
- Evaluates the trained model using the BLEU score metric.

## Requirements

- Python 3.6+
- PyTorch 1.7+
- Transformers library (Hugging Face)
- Sacrebleu library

## Dataset

The code assumes the existence of two files: `french_sentences.txt` and `english_sentences.txt`, containing parallel French and English sentences, respectively. You can replace these files with your own dataset.

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/ajliouat/machine-translation-with-transformers.git
   ```

2. Install the required dependencies:
   ```
   pip install torch transformers sacrebleu
   ```

3. Prepare your dataset:
   - Place your parallel French and English sentences in `french_sentences.txt` and `english_sentences.txt` files, respectively.

4. Set the hyperparameters:
   - Modify the hyperparameters in the code according to your requirements, such as the number of epochs, batch size, model dimensions, etc.

5. Run the code:
   ```
   python machine_translation.py
   ```

6. Monitor the training progress and evaluate the model:
   - The training loop will display the loss for each epoch.
   - After training, the model will be evaluated on the validation set, and the BLEU score will be reported.

## Model Architecture

The Transformer model consists of an encoder and a decoder, each composed of multiple layers. The encoder processes the source sequence (French), while the decoder generates the target sequence (English). The key components of the architecture are:

- Positional Encoding: Adds positional information to the input embeddings.
- Multi-Head Attention: Allows the model to attend to different parts of the input sequence.
- Feed-Forward Networks: Applies non-linear transformations to the attention outputs.
- Residual Connections and Layer Normalization: Facilitates training of deep models.

## Training

The model is trained using the AdamW optimizer with a linear learning rate scheduler and warmup. The training loop utilizes teacher forcing, where the decoder receives the ground truth target sequence as input during training. The loss is computed using cross-entropy loss, ignoring the padding tokens.

## Evaluation

The trained model is evaluated on a validation set using the BLEU score metric. The `sacrebleu` library is used to calculate the BLEU score, which measures the similarity between the generated translations and the reference translations.

## Acknowledgments

This implementation is based on the paper "Attention Is All You Need" by Vaswani et al. (2017) and draws inspiration from various open-source implementations and tutorials.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more information.

## Contact

For questions or feedback, please contact [a.jliouat@yahoo.fr](mailto:a.jliouat@yahoo.fr).
