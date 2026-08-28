#!/usr/bin/env python3
import argparse

from ink_ml import dry_run, result_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dependency-free Inkgraph RNN/DQN/GAN dry run.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dataset-size", type=int, default=24)
    args = parser.parse_args()
    if args.epochs < 1 or args.dataset_size < 1:
        parser.error("epochs and dataset-size must be positive")
    print(result_json(dry_run(args.epochs, args.dataset_size)))


if __name__ == "__main__":
    main()