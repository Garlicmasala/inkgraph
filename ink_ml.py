"""Small, dependency-free reference models for Inkgraph.

This is intentionally a dry-run implementation for validating the product
loop. Replace the scalar math with torch tensors for production training.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass
class InkExample:
    source: str
    family: str
    features: list[float]
    target: list[float]


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


def tanh(value: float) -> float:
    return math.tanh(value)


class InkRNN:
    """Tiny recurrent encoder that predicts three ink controls per source."""

    def __init__(self, input_size: int = 5, hidden_size: int = 8, seed: int = 11):
        rng = random.Random(seed)
        self.hidden_size = hidden_size
        self.weights = [[rng.uniform(-0.4, 0.4) for _ in range(input_size + hidden_size)] for _ in range(hidden_size)]
        self.output = [[rng.uniform(-0.4, 0.4) for _ in range(hidden_size)] for _ in range(3)]

    def encode(self, sequence: Iterable[list[float]]) -> list[float]:
        hidden = [0.0] * self.hidden_size
        for features in sequence:
            joined = features + hidden
            hidden = [tanh(sum(weight * value for weight, value in zip(row, joined))) for row in self.weights]
        return hidden

    def predict(self, features: list[float]) -> list[float]:
        hidden = self.encode([features])
        return [max(0.0, min(1.0, tanh(sum(weight * value for weight, value in zip(row, hidden))) / 2 + 0.5)) for row in self.output]

    def train_step(self, example: InkExample, learning_rate: float = 0.08) -> float:
        prediction = self.predict(example.features)
        errors = [target - actual for target, actual in zip(example.target, prediction)]
        for row, error in zip(self.output, errors):
            for index in range(len(row)):
                row[index] += learning_rate * error * 0.1
        return sum(error * error for error in errors) / len(errors)


class InkGAN:
    """Toy generator/discriminator pair for style parameter proposals."""

    def __init__(self, seed: int = 19):
        self.rng = random.Random(seed)
        self.generator_bias = [0.5, 0.2, 0.5]
        self.discriminator_bias = 0.5

    def generate(self, condition: list[float]) -> list[float]:
        return [max(0.0, min(1.0, bias + condition[index % len(condition)] * 0.1 + self.rng.uniform(-0.05, 0.05))) for index, bias in enumerate(self.generator_bias)]

    def train_step(self, example: InkExample, learning_rate: float = 0.05) -> float:
        generated = self.generate(example.features)
        loss = sum((real - fake) ** 2 for real, fake in zip(example.target, generated)) / len(generated)
        for index, target in enumerate(example.target):
            self.generator_bias[index] += learning_rate * (target - generated[index])
        return loss


class InkDQN:
    """Minimal Q learner for choosing the next graph-to-ink transform."""

    ACTIONS = ("preserve-layout", "add-wobble", "add-wash", "add-labels")

    def __init__(self, seed: int = 23):
        rng = random.Random(seed)
        self.q_values = [rng.uniform(0.0, 0.2) for _ in self.ACTIONS]

    def choose(self, epsilon: float = 0.0, rng: random.Random | None = None) -> str:
        generator = rng or random.Random(31)
        if generator.random() < epsilon:
            return generator.choice(self.ACTIONS)
        return self.ACTIONS[max(range(len(self.q_values)), key=self.q_values.__getitem__)]

    def train_step(self, reward: float, action: str = "preserve-layout", learning_rate: float = 0.2) -> float:
        index = self.ACTIONS.index(action)
        old_value = self.q_values[index]
        self.q_values[index] += learning_rate * (reward - old_value)
        return abs(reward - old_value)


def dry_run(epochs: int = 3, dataset_size: int = 24, seed: int = 7) -> dict:
    dataset = make_dataset(dataset_size, seed)
    rnn, gan, dqn = InkRNN(seed=seed + 1), InkGAN(seed=seed + 2), InkDQN(seed=seed + 3)
    history = []
    for epoch in range(1, epochs + 1):
        rnn_loss = sum(rnn.train_step(example) for example in dataset) / len(dataset)
        gan_loss = sum(gan.train_step(example) for example in dataset) / len(dataset)
        dqn_loss = dqn.train_step(0.9 if epoch > 1 else 0.6)
        history.append({"epoch": epoch, "rnn_loss": round(rnn_loss, 5), "gan_loss": round(gan_loss, 5), "dqn_loss": round(dqn_loss, 5)})
    sample = dataset[0]
    return {"dataset_size": len(dataset), "history": history, "sample": {"source": sample.source, "family": sample.family, "rnn_controls": [round(value, 3) for value in rnn.predict(sample.features)], "recommended_action": dqn.choose(), "gan_controls": [round(value, 3) for value in gan.generate(sample.features)]}}


def result_json(result: dict) -> str:
    return json.dumps(result, indent=2)