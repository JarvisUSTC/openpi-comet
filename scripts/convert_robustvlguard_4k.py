#!/usr/bin/env python

import argparse
import pathlib

import openpi.training.robustvlguard as robustvlguard


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert RobustVLGuard comprehensive 4k JSONL into the local_vqa_schema format."
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=pathlib.Path("/vepfs-C/dataset/RobustVLGuard/Extracted/comprehensive_4k_sft_gpt_anno.jsonl"),
        help="Path to the raw RobustVLGuard JSONL annotation file.",
    )
    parser.add_argument(
        "--image-root",
        type=pathlib.Path,
        default=pathlib.Path("/vepfs-C/dataset/RobustVLGuard/Extracted"),
        help="Root directory used to resolve each sample's relative image path.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("/vepfs-C/dataset/RobustVLGuard/Extracted/comprehensive_4k_openpi_schema.jsonl"),
        help="Output path for the converted local_vqa_schema JSONL.",
    )
    parser.add_argument(
        "--task-family",
        default=robustvlguard.DEFAULT_TASK_FAMILY,
        help="task_family field written to each converted sample.",
    )
    parser.add_argument(
        "--task-name",
        default=robustvlguard.DEFAULT_TASK_NAME,
        help="task_name field written to each converted sample.",
    )
    parser.add_argument(
        "--source",
        default=robustvlguard.DEFAULT_SOURCE,
        help="source field written to each converted sample.",
    )
    parser.add_argument(
        "--skip-image-validation",
        action="store_true",
        help="Do not verify that each referenced image path exists.",
    )
    args = parser.parse_args()

    converted = robustvlguard.convert_jsonl(
        args.input,
        args.output,
        image_root=args.image_root,
        task_family=args.task_family,
        task_name=args.task_name,
        source=args.source,
        validate_images=not args.skip_image_validation,
    )
    print(f"Converted {converted} samples to {args.output}")


if __name__ == "__main__":
    main()
