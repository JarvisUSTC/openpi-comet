#!/usr/bin/env python
import argparse
import logging
import os

import openpi.shared.download as _download
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def _iter_data_configs(config: _config.TrainConfig) -> list[_config.DataConfig]:
    factories = config.data if isinstance(config.data, list) else [config.data]
    return [factory.create(config.assets_dirs, config.model) for factory in factories]


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute reusable dataset caches for training.")
    parser.add_argument("config_name", help="Training config name from src/openpi/training/config.py")
    parser.add_argument(
        "--touch-first-sample",
        action="store_true",
        help="Also materialize the first sample of each dataset. Slower, but can catch path issues early.",
    )
    parser.add_argument(
        "--fast-tokenizer-path",
        help="Optional local FAST tokenizer directory for offline environments.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.fast_tokenizer_path:
        os.environ["OPENPI_FAST_TOKENIZER_PATH"] = args.fast_tokenizer_path

    config = _config.get_config(args.config_name)
    data_configs = _iter_data_configs(config)

    logging.info("Shared cache root: %s", _download.get_cache_dir())
    logging.info("HF datasets cache dir: %s", _download.get_hf_datasets_cache_dir())

    for idx, data_config in enumerate(data_configs):
        logging.info(
            "Precomputing dataset %s type=%s repo_id=%s",
            idx,
            data_config.dataset_type,
            data_config.repo_id,
        )
        if data_config.dataset_type == "behavior" and data_config.behavior_dataset_root:
            logging.info(
                "Behavior chunk cache dir for dataset %s: %s",
                idx,
                _download.get_behavior_cache_dir(data_config.behavior_dataset_root) / "chunks",
            )
        dataset = _data_loader.create_dataset(data_config, config.model, num_samples=max(config.batch_size, 1))
        logging.info("Dataset %s ready, len=%s", idx, len(dataset))
        if args.touch_first_sample and len(dataset) > 0:
            _ = dataset[0]
            logging.info("Materialized first sample for dataset %s", idx)


if __name__ == "__main__":
    main()
