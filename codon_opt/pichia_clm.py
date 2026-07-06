"""
Pichia-CLM Deep Learning Backend (TensorFlow)
==============================================

Encoder-decoder Bi-GRU with attention, trained on ~5000 K. phaffii CDS.
Reference: Narayanan et al., PNAS 2026, doi:10.1073/pnas.2522052123
Repository: https://github.com/NHarini-1995/PichiaCLM

STATUS: Weights incomplete in public repo (.ckpt.data files missing).
        Architecture is fully reconstructed here for when weights become available.
"""

import os
import logging
from typing import Dict, Optional

import numpy as np

from .config import (
    PICHIA_CLM_CHECKPOINT_DIR,
    PICHIA_CLM_HYPER_CSV,
    MAX_PROTEIN_LENGTH,
)

logger = logging.getLogger(__name__)

# Token dictionaries (from paper's notebook)
AA_DICT = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
    'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
    'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, 'X': 21, 'Z': 21,
    'B': 21, 'U': 21, 'O': 21, '*': 22,
}
AA_START = 24
AA_END = 23

CODON_LIST = [
    'GCT', 'GCC', 'GCA', 'GCG',          # A
    'TGT', 'TGC',                          # C
    'GAT', 'GAC',                          # D
    'GAA', 'GAG',                          # E
    'TTT', 'TTC',                          # F
    'GGT', 'GGA', 'GGC', 'GGG',          # G
    'CAT', 'CAC',                          # H
    'ATT', 'ATC', 'ATA',                  # I
    'AAA', 'AAG',                          # K
    'TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG',  # L
    'ATG',                                  # M
    'AAT', 'AAC',                          # N
    'CCT', 'CCC', 'CCA', 'CCG',          # P
    'CAA', 'CAG',                          # Q
    'CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG',  # R
    'TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC',  # S
    'ACT', 'ACC', 'ACA', 'ACG',          # T
    'GTT', 'GTC', 'GTA', 'GTG',          # V
    'TGG',                                  # W
    'TAT', 'TAC',                          # Y
    'TAA', 'TAG', 'TGA',                  # *
]
CODON_DICT = {c: i + 1 for i, c in enumerate(CODON_LIST)}
CODON_START = 65
CODON_END = 66

# AA token -> valid codon tokens mapping (for constrained decoding)
AA_TO_CODON_TOKENS = {}
_aa_for_codons = [
    'A', 'A', 'A', 'A', 'C', 'C', 'D', 'D', 'E', 'E',
    'F', 'F', 'G', 'G', 'G', 'G', 'H', 'H', 'I', 'I', 'I',
    'K', 'K', 'L', 'L', 'L', 'L', 'L', 'L', 'M',
    'N', 'N', 'P', 'P', 'P', 'P', 'Q', 'Q',
    'R', 'R', 'R', 'R', 'R', 'R', 'S', 'S', 'S', 'S', 'S', 'S',
    'T', 'T', 'T', 'T', 'V', 'V', 'V', 'V', 'W',
    'Y', 'Y', '*', '*', '*',
]
for idx, aa in enumerate(_aa_for_codons):
    token_aa = AA_DICT.get(aa, 22)
    if token_aa not in AA_TO_CODON_TOKENS:
        AA_TO_CODON_TOKENS[token_aa] = []
    AA_TO_CODON_TOKENS[token_aa].append(idx + 1)
# Map end tokens
AA_TO_CODON_TOKENS[AA_END] = [CODON_END]


def _check_weights() -> bool:
    """Check if Pichia-CLM checkpoint data files exist."""
    data_file = os.path.join(PICHIA_CLM_CHECKPOINT_DIR, "FinArch1_AttnCorr_cp.ckpt.data-00000-of-00001")
    return os.path.isfile(data_file)


def _build_model():
    """Reconstruct Pichia-CLM Arch1 model from hyperparameters."""
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Input, Embedding, Bidirectional, GRU, Concatenate,
        Attention, TimeDistributed, Dense, Dropout,
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from keras.losses import sparse_categorical_crossentropy
    import pandas as pd

    # Load hyperparameters (Setting_no=1 from Round3.csv, as in the notebook)
    settings = pd.read_csv(PICHIA_CLM_HYPER_CSV).iloc[:, 1:]
    setting_no = 1

    max_length = 1000
    aa_vocab_size = 25
    dna_vocab_size = 67

    hidden_enc = int(settings['Enc hidden size'][setting_no])
    emb_enc = int(settings['Enc Embedding size'][setting_no])
    emb_dec = int(settings['Dec Embedding size'][setting_no])
    dense_size = int(settings['Dense Layer size'][setting_no])
    dense_size_aa = int(settings['Dense Layer size aa'][setting_no])
    drop_rate = float(settings['Drop rate'][setting_no])
    drop_rate_aa = float(settings['Drop rate aa'][setting_no])

    # Encoder
    input_sequence = Input(shape=(max_length,), name='encoder_input')
    encod_emb = Embedding(aa_vocab_size, emb_enc, mask_zero=True, name='aa_embedding')
    embedding = encod_emb(input_sequence)
    encoder = Bidirectional(
        GRU(hidden_enc, return_sequences=True, return_state=True),
        merge_mode="concat"
    )
    enc_seq, enc_final_f, enc_final_b = encoder(embedding)
    enc_final = Concatenate(axis=-1)([enc_final_f, enc_final_b])

    # Decoder (codon)
    decoder_inputs = Input(shape=(max_length - 1,), name='decoder_codon_input')
    decoder_inputs_aa = Input(shape=(max_length,), name='decoder_aa_input')

    dex = Embedding(dna_vocab_size, emb_dec, mask_zero=True, name='codon_embedding')
    final_dex = dex(decoder_inputs)
    final_dex_aa = encod_emb(decoder_inputs_aa)

    decoder = GRU(2 * hidden_enc, return_sequences=True, return_state=True, name='decoder_gru')
    decoder_aa = GRU(2 * hidden_enc, return_sequences=True, return_state=True, name='decoder_aa_gru')

    dec_seq, _ = decoder(final_dex, initial_state=enc_final)
    dec_seq_aa, _ = decoder_aa(final_dex_aa, initial_state=enc_final)

    attn_layer = Attention(name='attention')
    attn_out = attn_layer([dec_seq, enc_seq])
    attn_layer_aa = Attention(name='attention_aa')
    attn_out_aa = attn_layer_aa([dec_seq_aa, enc_seq])

    dec_concat = Concatenate(axis=-1)([dec_seq, attn_out])
    dec_concat_aa = Concatenate(axis=-1)([dec_seq_aa, attn_out_aa])

    inter_layer = TimeDistributed(Dense(dense_size, activation='tanh'), name='intermediate')
    inter_layer_aa = TimeDistributed(Dense(dense_size_aa, activation='tanh'), name='intermediate_aa')

    inter_out = inter_layer(dec_concat)
    inter_out_aa = inter_layer_aa(dec_concat_aa)

    dropout_layer = Dropout(drop_rate)
    dropout_out = dropout_layer(inter_out)
    dropout_out_aa = dropout_layer(inter_out_aa)

    dense_layer = TimeDistributed(Dense(dna_vocab_size, activation='softmax'), name='output_codon')
    logits = dense_layer(dropout_out)

    dense_layer_aa = TimeDistributed(Dense(aa_vocab_size, activation='softmax'), name='output_aa')
    logits_aa = dense_layer_aa(dropout_out_aa)

    # Full model
    full_model = Model(
        [input_sequence, decoder_inputs, decoder_inputs_aa],
        [logits, logits_aa]
    )
    full_model.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    # Load weights
    ckpt_path = os.path.join(PICHIA_CLM_CHECKPOINT_DIR, "FinArch1_AttnCorr_cp.ckpt")
    full_model.load_weights(ckpt_path)

    # Build inference models
    encoder_model = Model(input_sequence, [enc_final, enc_seq])

    # Decoder inference (step-by-step)
    inf_enc_seq = Input(shape=(max_length, 2 * hidden_enc), name='inf_enc_seq')
    inf_state = Input(shape=(2 * hidden_enc,), name='inf_state')
    inf_codon_in = Input(shape=(1,), name='inf_codon_in')

    inf_dex = dex(inf_codon_in)
    inf_dec_out, inf_dec_state = decoder(inf_dex, initial_state=inf_state)
    inf_attn = attn_layer([inf_dec_out, inf_enc_seq])
    inf_concat = Concatenate(axis=-1)([inf_dec_out, inf_attn])
    inf_logits = dense_layer(dropout_layer(inter_layer(inf_concat)))

    decoder_model = Model(
        [inf_codon_in, inf_state, inf_enc_seq],
        [inf_logits, inf_dec_state]
    )

    return encoder_model, decoder_model


# Cached models
_encoder_model = None
_decoder_model = None


def _get_models():
    global _encoder_model, _decoder_model
    if _encoder_model is None:
        _encoder_model, _decoder_model = _build_model()
    return _encoder_model, _decoder_model


def tokenize_protein(protein_seq: str) -> np.ndarray:
    """Tokenize protein sequence for Pichia-CLM input."""
    seq = protein_seq.upper().strip()
    if seq.endswith('*'):
        pass
    else:
        seq = seq + '*'

    tokens = [AA_START]
    for aa in seq:
        tokens.append(AA_DICT.get(aa, 21))
    tokens.append(AA_END)

    # Pad to 1001
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    padded = pad_sequences([tokens], maxlen=1001, dtype='int32', padding='post', truncating='post')
    return padded


def detokenize_codons(codon_tokens: list) -> str:
    """Convert codon token sequence back to DNA string."""
    dna = ''
    for t in codon_tokens:
        if t == CODON_START or t == CODON_END or t == 0:
            continue
        if 1 <= t <= len(CODON_LIST):
            dna += CODON_LIST[t - 1]
    return dna


def predict_optimized_dna(protein_seq: str, add_stop: bool = True) -> Dict:
    """
    Run Pichia-CLM inference on a protein sequence.

    Returns dict with dna_sequence or error.
    """
    if not _check_weights():
        return {
            'error': 'Pichia-CLM weights not found',
            'hint': (
                'Checkpoint .data files missing from '
                f'{PICHIA_CLM_CHECKPOINT_DIR}. '
                'Contact authors (Narayanan et al., PNAS 2026) for weights, '
                'or use method="dnachisel" instead.'
            ),
        }

    try:
        encoder_model, decoder_model = _get_models()
    except Exception as e:
        return {'error': f'Failed to build Pichia-CLM model: {e}'}

    # Tokenize
    padded = tokenize_protein(protein_seq)
    input_seq = padded[:, 1:MAX_PROTEIN_LENGTH + 1]

    # Encode
    final_state, enc_sequence = encoder_model.predict(input_seq, verbose=0)

    # Autoregressive decode with AA-constrained masking
    dna_vocab = 67
    target = np.zeros((1, 1))
    target[0, 0] = CODON_START
    states = final_state
    decoded = [CODON_START]

    for pos in range(MAX_PROTEIN_LENGTH):
        aa_token = input_seq[0, pos]
        if aa_token == 0:  # padding
            break

        output, states = decoder_model.predict([target, states, enc_sequence], verbose=0)

        # Mask: only allow codons for this amino acid
        mask = np.zeros((1, dna_vocab))
        valid_codons = AA_TO_CODON_TOKENS.get(int(aa_token), [])
        if valid_codons:
            mask[0, valid_codons] = 1.0
        masked = output[0] * mask
        token = int(np.argmax(masked))

        decoded.append(token)
        if token == CODON_END:
            break
        target[0, 0] = token

    dna = detokenize_codons(decoded)

    result = {
        'dna_sequence': dna,
        'protein_sequence': protein_seq.upper().strip(),
        'method': 'pichia_clm',
        'organism': 'Komagataella phaffii',
        'length_nt': len(dna),
        'length_aa': len(protein_seq.strip()),
    }

    if not add_stop and dna.endswith(('TAA', 'TAG', 'TGA')):
        result['dna_sequence'] = dna[:-3]
        result['length_nt'] -= 3

    return result
