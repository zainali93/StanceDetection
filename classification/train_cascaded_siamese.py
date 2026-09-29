"""
Cascaded Siamese BERT-LSTM-Attention model for stance detection.

Paper:
    Enhanced Stance Detection using Cascaded Siamese Networks
    with Attention Mechanism (ICONIP 2024)

The four-class FNC-1 stance detection problem is decomposed into:

    Model 1: Related vs. Unrelated
    Model 2: Agree vs. Discuss vs. Disagree

Model 2 is applied only to examples predicted as related by Model 1.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, TFAutoModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "bert-base-uncased"

DATA_DIR = Path("data")

TRAIN_BODIES = DATA_DIR / "train_bodies.csv"
TRAIN_STANCES = DATA_DIR / "train_stances.csv"
TEST_BODIES = DATA_DIR / "competition_test_bodies.csv"
TEST_STANCES = DATA_DIR / "competition_test_stances.csv"

HEADLINE_MAX_LENGTH = 15
BODY_MAX_LENGTH = 300

LSTM_UNITS = 128
DROPOUT_RATE = 0.5

LEARNING_RATE = 2e-3
BATCH_SIZE = 20
EPOCHS = 100
VALIDATION_SPLIT = 0.20

UNRELATED_SAMPLES = 25000

# Threshold used in binary relevance classifier.
RELEVANCE_THRESHOLD = 0.01

SEED = 42

CATEGORY_TO_ID = {
    "agree": 0,
    "discuss": 1,
    "disagree": 2,
}

ID_TO_CATEGORY = {
    0: "agree",
    1: "discuss",
    2: "disagree",
}

STANCE_LABELS = [
    "agree",
    "disagree",
    "discuss",
    "unrelated",
]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_fnc_data():
    """Load and merge the official FNC-1 train and test files."""

    required_files = [
        TRAIN_BODIES,
        TRAIN_STANCES,
        TEST_BODIES,
        TEST_STANCES,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset file not found: {path}"
            )

    train_bodies = pd.read_csv(TRAIN_BODIES)
    train_stances = pd.read_csv(TRAIN_STANCES)

    test_bodies = pd.read_csv(TEST_BODIES)
    test_stances = pd.read_csv(TEST_STANCES)

    train = train_stances.merge(
        train_bodies,
        on="Body ID",
        how="left",
        validate="many_to_one",
    )

    test = test_stances.merge(
        test_bodies,
        on="Body ID",
        how="left",
        validate="many_to_one",
    )

    if train["articleBody"].isna().any():
        raise ValueError(
            "Some training stances could not be matched to article bodies."
        )

    if test["articleBody"].isna().any():
        raise ValueError(
            "Some test stances could not be matched to article bodies."
        )

    return train, test


def prepare_relevance_data(train):
    """
    Prepare Model 1 data.

    Related:
        agree, discuss, disagree -> 1

    Unrelated:
        unrelated -> 0

    The unrelated class is randomly undersampled to 25,000 examples.
    """

    unrelated = train[
        train["Stance"] == "unrelated"
    ].sample(
        n=UNRELATED_SAMPLES,
        random_state=SEED,
        replace=False,
    )

    related = train[
        train["Stance"] != "unrelated"
    ]

    data = pd.concat(
        [unrelated, related],
        ignore_index=True,
    )

    data["label"] = (
        data["Stance"] != "unrelated"
    ).astype(np.float32)

    data = data.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    return data


def prepare_category_data(train):
    """
    Prepare Model 2 data.

    Only related examples are retained:

        agree    -> 0
        discuss  -> 1
        disagree -> 2
    """

    data = train[
        train["Stance"] != "unrelated"
    ].copy()

    data["label"] = data[
        "Stance"
    ].map(CATEGORY_TO_ID)

    data = data.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    return data


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_pairs(tokenizer, data):
    """Tokenize headlines and article bodies separately."""

    headlines = tokenizer(
        data["Headline"].astype(str).tolist(),
        padding=True,
        truncation=True,
        max_length=HEADLINE_MAX_LENGTH,
        return_tensors="tf",
    )

    bodies = tokenizer(
        data["articleBody"].astype(str).tolist(),
        padding=True,
        truncation=True,
        max_length=BODY_MAX_LENGTH,
        return_tensors="tf",
    )

    return (
        batch_encoding_to_tuple(headlines),
        batch_encoding_to_tuple(bodies),
    )


def batch_encoding_to_tuple(encoded):
    """
    Hugging Face TensorFlow BERT receives:
        input_ids,
        token_type_ids,
        attention_mask
    """

    return (
        encoded["input_ids"].numpy(),
        encoded["token_type_ids"].numpy(),
        encoded["attention_mask"].numpy(),
    )


def subset_inputs(inputs, indices):
    """Select a subset of tokenized BERT inputs."""

    return tuple(
        array[indices]
        for array in inputs
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SiameseBERTLSTMAttentionCategory(tf.keras.Model):
    """
    Three-class Siamese model used for Model 2.

    The BERT encoder and LSTM are shared between the headline and
    article-body branches.
    """

    def __init__(self, bert_model):
        super().__init__()

        self.bert = bert_model

        self.lstm = tf.keras.layers.LSTM(
            LSTM_UNITS,
            return_sequences=True,
        )

        self.attention = tf.keras.layers.Attention()

        self.dropout = tf.keras.layers.Dropout(
            DROPOUT_RATE
        )

        self.fc = tf.keras.layers.Dense(
            3,
            activation="softmax",
        )

    def call(self, inputs, training=False):

        headline, body = inputs

        headline = self.bert(
            headline,
            training=False,
        )[0]

        body = self.bert(
            body,
            training=False,
        )[0]

        headline = self.lstm(
            headline,
            training=training,
        )

        body = self.lstm(
            body,
            training=training,
        )

        context = self.attention(
            [headline, body]
        )

        context = self.dropout(
            context,
            training=training,
        )

        context = tf.keras.layers.Flatten()(
            context
        )

        return self.fc(context)


class SiameseBERTLSTMAttentionRelevance(tf.keras.Model):
    """
    Binary Siamese model used for Model 1.
    """

    def __init__(self, bert_model):
        super().__init__()

        self.bert = bert_model

        self.lstm = tf.keras.layers.LSTM(
            LSTM_UNITS,
            return_sequences=True,
        )

        self.attention = tf.keras.layers.Attention()

        self.dropout = tf.keras.layers.Dropout(
            DROPOUT_RATE
        )

        self.fc = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
        )

    def call(self, inputs, training=False):

        headline, body = inputs

        headline = self.bert(
            headline,
            training=False,
        )[0]

        body = self.bert(
            body,
            training=False,
        )[0]

        headline = self.lstm(
            headline,
            training=training,
        )

        body = self.lstm(
            body,
            training=training,
        )

        context = self.attention(
            [headline, body]
        )

        context = self.dropout(
            context,
            training=training,
        )

        context = tf.keras.layers.Flatten()(
            context
        )

        return self.fc(context)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def load_frozen_bert():
    """Load a frozen bert-base-uncased encoder."""

    bert = TFAutoModel.from_pretrained(
        MODEL_NAME
    )

    for layer in bert.layers:
        layer.trainable = False

    return bert


def best_model_callback():
    """
    Track validation loss throughout all epochs and restore the
    weights corresponding to the lowest validation loss.

    Patience is set equal to the total number of epochs so that
    training runs for the full configured duration.
    """

    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=EPOCHS,
        restore_best_weights=True,
        verbose=1,
    )


def train_relevance_model(
    tokenizer,
    train,
):
    """Train Model 1: related vs. unrelated."""

    print("\n" + "=" * 70)
    print("MODEL 1: RELATED VS. UNRELATED")
    print("=" * 70)

    data = prepare_relevance_data(train)

    print("\nTraining distribution:")
    print(
        data["Stance"]
        .value_counts()
    )

    headline_inputs, body_inputs = tokenize_pairs(
        tokenizer,
        data,
    )

    labels = data[
        "label"
    ].to_numpy(dtype=np.float32)

    bert = load_frozen_bert()

    model = SiameseBERTLSTMAttentionRelevance(
        bert
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(
        x=[
            headline_inputs,
            body_inputs,
        ],
        y=labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        shuffle=True,
        callbacks=[best_model_callback()],
    )

    return model


def train_category_model(
    tokenizer,
    train,
):
    """Train Model 2: agree vs. discuss vs. disagree."""

    print("\n" + "=" * 70)
    print("MODEL 2: AGREE / DISCUSS / DISAGREE")
    print("=" * 70)

    data = prepare_category_data(train)

    print("\nTraining distribution:")
    print(
        data["Stance"]
        .value_counts()
    )

    headline_inputs, body_inputs = tokenize_pairs(
        tokenizer,
        data,
    )

    labels = tf.keras.utils.to_categorical(
        data["label"].to_numpy(),
        num_classes=3,
    )

    bert = load_frozen_bert()

    model = SiameseBERTLSTMAttentionCategory(
        bert
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(
        x=[
            headline_inputs,
            body_inputs,
        ],
        y=labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        shuffle=True,
        callbacks=[best_model_callback()],
    )

    return model


# ---------------------------------------------------------------------------
# Cascaded inference
# ---------------------------------------------------------------------------

def predict_cascade(
    tokenizer,
    relevance_model,
    category_model,
    test,
):
    """
    Run the two-stage classifier.

    Model 1 first predicts whether each example is related.
    Model 2 is then applied only to examples predicted as related.
    """

    headline_inputs, body_inputs = tokenize_pairs(
        tokenizer,
        test,
    )

    relevance_probabilities = (
        relevance_model.predict(
            [
                headline_inputs,
                body_inputs,
            ],
            batch_size=BATCH_SIZE,
            verbose=1,
        ).reshape(-1)
    )

    related_mask = (
        relevance_probabilities
        > RELEVANCE_THRESHOLD
    )

    predictions = np.full(
        len(test),
        "unrelated",
        dtype=object,
    )

    related_indices = np.where(
        related_mask
    )[0]

    print(
        f"\nPredicted related: "
        f"{len(related_indices):,}"
    )

    print(
        f"Predicted unrelated: "
        f"{len(test) - len(related_indices):,}"
    )

    if len(related_indices) == 0:
        return predictions

    related_headlines = subset_inputs(
        headline_inputs,
        related_indices,
    )

    related_bodies = subset_inputs(
        body_inputs,
        related_indices,
    )

    category_probabilities = (
        category_model.predict(
            [
                related_headlines,
                related_bodies,
            ],
            batch_size=BATCH_SIZE,
            verbose=1,
        )
    )

    category_ids = np.argmax(
        category_probabilities,
        axis=1,
    )

    category_predictions = np.array(
        [
            ID_TO_CATEGORY[index]
            for index in category_ids
        ]
    )

    predictions[
        related_indices
    ] = category_predictions

    return predictions


# ---------------------------------------------------------------------------
# FNC evaluation metric
# ---------------------------------------------------------------------------

def fnc_score(gold_labels, predicted_labels):
    """
    Compute the official FNC-1 weighted score.

    Scoring:
        +0.25 for correctly identifying relatedness.
        +0.50 for the correct related stance category.
        +0.25 additional credit when both gold and prediction
        are in the related category.
    """

    related = {
        "agree",
        "disagree",
        "discuss",
    }

    score = 0.0
    max_score = 0.0

    for gold, prediction in zip(
        gold_labels,
        predicted_labels,
    ):

        if gold == "unrelated":
            max_score += 0.25
        else:
            max_score += 1.0

        if gold == prediction:
            score += 0.25

            if gold != "unrelated":
                score += 0.50

        if (
            gold in related
            and prediction in related
        ):
            score += 0.25

    return score, max_score


def evaluate(test, predictions):
    """Report class-wise F1, accuracy, and the FNC metric."""

    gold = test[
        "Stance"
    ].astype(str).to_numpy()

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(
        classification_report(
            gold,
            predictions,
            labels=STANCE_LABELS,
            digits=4,
            zero_division=0,
        )
    )

    accuracy = accuracy_score(
        gold,
        predictions,
    )

    score, max_score = fnc_score(
        gold,
        predictions,
    )

    normalized_score = (
        score / max_score
        if max_score
        else 0.0
    )

    print(
        f"Accuracy:   {accuracy:.4f}"
    )

    print(
        f"FNC score:  "
        f"{score:.2f} / {max_score:.2f}"
    )

    print(
        f"FNC metric: {normalized_score:.4f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    print("=" * 70)
    print(
        "Cascaded Siamese Network for FNC-1 Stance Detection"
    )
    print("=" * 70)

    train, test = load_fnc_data()

    print(
        f"Training examples: {len(train):,}"
    )

    print(
        f"Test examples:     {len(test):,}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    relevance_model = train_relevance_model(
        tokenizer,
        train,
    )

    category_model = train_category_model(
        tokenizer,
        train,
    )

    predictions = predict_cascade(
        tokenizer,
        relevance_model,
        category_model,
        test,
    )

    evaluate(
        test,
        predictions,
    )


if __name__ == "__main__":
    main()
