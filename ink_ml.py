"""Small, dependency-free reference models for Inkgraph.

This is intentionally a dry-run implementation for validating the product
loop. Replace the scalar math with torch tensors for production training.
"""
from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Iterable


@dataclass
class InkExample:
    source: str
    family: str
    features: list[float]
    target: list[float]

    def __post_init__(self):
        if not self.source.strip() or self.family not in STYLE_TARGETS:
            raise ValueError("InkExample requires a source and known style family")
        if len(self.features) != 5 or len(self.target) != 3 or not all(0 <= value <= 1 for value in self.target):
            raise ValueError("InkExample feature/target dimensions are invalid")


STYLE_TARGETS = {
    "sumi": [0.78, 0.22, 0.62],
    "field-notes": [0.42, 0.08, 0.28],
    "dry-brush": [0.66, 0.38, 0.84],
}


def tokenize_source(source: str) -> list[float]:
    """Encode a graph prompt into stable, model-friendly scalar features."""
    words = source.lower().split()
    arrows = source.count("→") + source.count("->")
    return [
        min(len(words), 32) / 32,
        min(len(source), 500) / 500,
        min(arrows, 8) / 8,
        sum(char.isdigit() for char in source) / max(len(source), 1),
        sum(char in ":,;" for char in source) / max(len(source), 1),
    ]


def make_dataset(size: int = 24, seed: int = 7) -> list[InkExample]:
    """Create a reproducible starter dataset when no user dataset is supplied."""
    if size < 1:
        raise ValueError("dataset size must be positive")
    rng = random.Random(seed)
    prompts = [
        "wake -> coffee -> focus -> walk",
        "research -> sketch -> prototype -> share",
        "input -> transform -> output",
        "observe -> practice -> reflect",
        "idea -> draft -> revise -> publish",
    ]
    families = list(STYLE_TARGETS)
    examples = []
    for index in range(size):
        source = prompts[index % len(prompts)]
        family = families[index % len(families)]
        features = tokenize_source(source)
        noise = [rng.uniform(-0.03, 0.03) for _ in range(3)]
        target = [max(0.0, min(1.0, value + noise[i])) for i, value in enumerate(STYLE_TARGETS[family])]
        examples.append(InkExample(source, family, features, target))
    return examples


def split_dataset(dataset: list[InkExample], validation_ratio: float = 0.2) -> tuple[list[InkExample], list[InkExample]]:
    """Make a deterministic holdout so dry runs report generalization, not only fit loss."""
    if not 0 < validation_ratio < 1 or len(dataset) < 2:
        raise ValueError("validation ratio must be between 0 and 1 and dataset must have at least two examples")
    split = max(1, min(len(dataset) - 1, round(len(dataset) * (1 - validation_ratio))))
    return dataset[:split], dataset[split:]


def load_jsonl_dataset(path: str) -> list[InkExample]:
    """Load licensed source/style pairs without coupling training to a vendor format."""
    examples = []
    with open(path, encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            source, family = str(record["source"]), str(record["family"])
            target = [float(value) for value in record.get("target", STYLE_TARGETS[family])]
            examples.append(InkExample(source, family, tokenize_source(source), target))
    if not examples:
        raise ValueError(f"dataset {path} contains no examples")
    return examples


def tanh(value: float) -> float:
    return math.tanh(value)


class InkRNN:
    """Tiny recurrent encoder that predicts three ink controls per source."""

    def __init__(self, input_size: int = 5, hidden_size: int = 8, seed: int = 11):
        rng = random.Random(seed)
        self.hidden_size = hidden_size
        self.weights = [[rng.uniform(-0.4, 0.4) for _ in range(input_size + hidden_size)] for _ in range(hidden_size)]
        self.bias = [0.0] * hidden_size
        self.output = [[rng.uniform(-0.4, 0.4) for _ in range(hidden_size)] for _ in range(3)]
        self.output_bias = [0.0] * 3

    def encode(self, sequence: Iterable[list[float]]) -> list[float]:
        hidden = [0.0] * self.hidden_size
        for features in sequence:
            joined = features + hidden
            hidden = [tanh(sum(weight * value for weight, value in zip(row, joined)) + self.bias[index]) for index, row in enumerate(self.weights)]
        return hidden

    def predict(self, features: list[float]) -> list[float]:
        hidden = self.encode([features])
        return [max(0.0, min(1.0, tanh(sum(weight * value for weight, value in zip(row, hidden)) + self.output_bias[index]) / 2 + 0.5)) for index, row in enumerate(self.output)]

    def train_step(self, example: InkExample, learning_rate: float = 0.08) -> float:
        hidden = self.encode([example.features])
        prediction = [max(0.0, min(1.0, tanh(sum(weight * value for weight, value in zip(row, hidden)) + self.output_bias[index]) / 2 + 0.5)) for index, row in enumerate(self.output)]
        errors = [target - actual for target, actual in zip(example.target, prediction)]
        for output_index, (row, error) in enumerate(zip(self.output, errors)):
            for index in range(len(row)):
                row[index] += learning_rate * error * hidden[index] * 0.5
            self.output_bias[output_index] += learning_rate * error * 0.5
        return sum(error * error for error in errors) / len(errors)


class InkGAN:
    """Toy generator/discriminator pair for style parameter proposals."""

    def __init__(self, seed: int = 19):
        self.rng = random.Random(seed)
        self.generator_bias = [0.5, 0.2, 0.5]
        self.discriminator_bias = 0.5
        self.discriminator_weights = [0.1, 0.1, 0.1]

    def generate(self, condition: list[float]) -> list[float]:
        return [max(0.0, min(1.0, bias + condition[index % len(condition)] * 0.1 + self.rng.uniform(-0.05, 0.05))) for index, bias in enumerate(self.generator_bias)]

    def discriminate(self, sample: list[float]) -> float:
        score = self.discriminator_bias + sum(weight * value for weight, value in zip(self.discriminator_weights, sample))
        return 1 / (1 + math.exp(-score))

    def train_step(self, example: InkExample, learning_rate: float = 0.05) -> float:
        generated = self.generate(example.features)
        real_score, fake_score = self.discriminate(example.target), self.discriminate(generated)
        reconstruction_loss = sum((real - fake) ** 2 for real, fake in zip(example.target, generated)) / len(generated)
        discriminator_loss = (1 - real_score) ** 2 + fake_score ** 2
        for index, target in enumerate(example.target):
            self.generator_bias[index] += learning_rate * (target - generated[index]) + learning_rate * (1 - fake_score) * 0.02
            self.discriminator_weights[index] += learning_rate * ((target - generated[index]) * 0.05)
        self.discriminator_bias += learning_rate * ((1 - real_score) - fake_score) * 0.05
        return reconstruction_loss + discriminator_loss * 0.1


class InkDQN:
    """Minimal Q learner for choosing the next graph-to-ink transform."""

    ACTIONS = ("preserve-layout", "add-wobble", "add-wash", "add-labels")

    def __init__(self, seed: int = 23, replay_capacity: int = 32, discount: float = 0.9):
        rng = random.Random(seed)
        self.q_values = [rng.uniform(0.0, 0.2) for _ in self.ACTIONS]
        self.target_values = self.q_values.copy()
        self.replay = deque(maxlen=replay_capacity)
        self.discount = discount
        self.rng = rng

    def choose(self, epsilon: float = 0.0, rng: random.Random | None = None) -> str:
        generator = rng or random.Random(31)
        if generator.random() < epsilon:
            return generator.choice(self.ACTIONS)
        return self.ACTIONS[max(range(len(self.q_values)), key=self.q_values.__getitem__)]

    def train_step(self, reward: float, action: str = "preserve-layout", learning_rate: float = 0.2) -> float:
        index = self.ACTIONS.index(action)
        self.replay.append((index, reward))
        old_value = self.q_values[index]
        target = reward + self.discount * max(self.target_values)
        self.q_values[index] += learning_rate * (target - old_value)
        if len(self.replay) % 4 == 0:
            self.target_values = [(target_value + q_value) / 2 for target_value, q_value in zip(self.target_values, self.q_values)]
        return abs(target - old_value)


def mean_squared_error(predictions: list[list[float]], targets: list[list[float]]) -> float:
    values = [difference ** 2 for prediction, target in zip(predictions, targets) for difference in (actual - expected for actual, expected in zip(prediction, target))]
    return sum(values) / max(1, len(values))


def evaluate_rnn(model: InkRNN, dataset: list[InkExample]) -> float:
    return mean_squared_error([model.predict(example.features) for example in dataset], [example.target for example in dataset])


def dry_run(epochs: int = 3, dataset_size: int = 24, seed: int = 7) -> dict:
    dataset = make_dataset(dataset_size, seed)
    training, validation = split_dataset(dataset)
    rnn, gan, dqn = InkRNN(seed=seed + 1), InkGAN(seed=seed + 2), InkDQN(seed=seed + 3)
    history = []
    for epoch in range(1, epochs + 1):
        rnn_loss = sum(rnn.train_step(example) for example in training) / len(training)
        gan_loss = sum(gan.train_step(example) for example in training) / len(training)
        dqn_loss = dqn.train_step(0.9 if epoch > 1 else 0.6, InkDQN.ACTIONS[epoch % len(InkDQN.ACTIONS)])
        history.append({"epoch": epoch, "rnn_loss": round(rnn_loss, 5), "gan_loss": round(gan_loss, 5), "dqn_loss": round(dqn_loss, 5), "validation_loss": round(evaluate_rnn(rnn, validation), 5)})
    sample = dataset[0]
    return {"dataset_size": len(dataset), "training_size": len(training), "validation_size": len(validation), "history": history, "sample": {"source": sample.source, "family": sample.family, "rnn_controls": [round(value, 3) for value in rnn.predict(sample.features)], "recommended_action": dqn.choose(), "gan_controls": [round(value, 3) for value in gan.generate(sample.features)]}}


def result_json(result: dict) -> str:
    return json.dumps(result, indent=2)