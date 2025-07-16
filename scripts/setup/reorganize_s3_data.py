#!/usr/bin/env python3
"""
S3 Data Reorganization Script

This script reorganizes existing data in S3 bucket into the proper YOLO directory structure
for training. It can handle various input formats and create the expected structure:

Expected Output Structure:
data/drone-detection/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/ (optional)
    ├── images/
    └── labels/

Usage:
    python scripts/setup/reorganize_s3_data.py --source-prefix raw-data/ --target-prefix data/drone-detection/
    python scripts/setup/reorganize_s3_data.py --analyze-only  # Just analyze current structure
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Tuple, Optional
import random
from pathlib import Path

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from data.s3_utils import S3DataAccess
from configs.project_config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class S3DataReorganizer:
    """
    Reorganizes S3 data into proper YOLO directory structure.
    """

    def __init__(self, bucket_name: str, aws_profile: str = "ab"):
        """
        Initialize the S3 data reorganizer.

        Args:
            bucket_name: S3 bucket name
            aws_profile: AWS profile to use
        """
        self.bucket_name = bucket_name
        self.aws_profile = aws_profile

        try:
            self.s3_client = S3DataAccess(bucket_name, aws_profile)
            logger.info(f"Initialized S3 client for bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            raise

    def analyze_current_structure(self, prefix: str = "") -> Dict[str, any]:
        """
        Analyze the current S3 bucket structure.

        Args:
            prefix: S3 prefix to analyze

        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing S3 structure with prefix: '{prefix}'")

        try:
            # List all objects without limit for large datasets
            objects = self.s3_client.list_objects(prefix=prefix, max_keys=None)

            analysis = {
                "total_objects": len(objects),
                "file_types": {},
                "directory_structure": {},
                "image_files": [],
                "label_files": [],
                "other_files": [],
                "potential_splits": set(),
                "class_directories": {},
                "images_by_class": {},
            }

            # Analyze each object with progress tracking
            total_objects = len(objects)
            logger.info(f"Analyzing {total_objects} objects...")

            for i, obj in enumerate(objects):
                key = obj["Key"]

                # Extract file extension
                if "." in key:
                    ext = key.split(".")[-1].lower()
                    analysis["file_types"][ext] = analysis["file_types"].get(ext, 0) + 1

                # Categorize files
                if self._is_image_file(key):
                    analysis["image_files"].append(key)

                    # Extract class from directory structure (first directory is class name)
                    path_parts = key.split("/")
                    if len(path_parts) > 1:
                        class_name = path_parts[0]  # First directory is class name
                        if class_name not in analysis["images_by_class"]:
                            analysis["images_by_class"][class_name] = []
                        analysis["images_by_class"][class_name].append(key)
                        analysis["class_directories"][class_name] = (
                            analysis["class_directories"].get(class_name, 0) + 1
                        )

                elif self._is_label_file(key):
                    analysis["label_files"].append(key)
                else:
                    analysis["other_files"].append(key)

                # Analyze directory structure
                path_parts = key.split("/")
                if len(path_parts) > 1:
                    for j, part in enumerate(path_parts[:-1]):  # Exclude filename
                        level = f"level_{j}"
                        if level not in analysis["directory_structure"]:
                            analysis["directory_structure"][level] = set()
                        analysis["directory_structure"][level].add(part)

                        # Look for potential train/val/test splits
                        if part.lower() in ["train", "val", "test", "validation"]:
                            analysis["potential_splits"].add(part.lower())

                # Progress tracking for large datasets
                if (i + 1) % 10000 == 0:
                    logger.info(
                        f"Analyzed {i + 1}/{total_objects} objects ({((i + 1)/total_objects)*100:.1f}%)"
                    )

            logger.info(f"Analysis completed for {total_objects} objects")

            # Convert sets to lists for JSON serialization
            for level in analysis["directory_structure"]:
                analysis["directory_structure"][level] = list(
                    analysis["directory_structure"][level]
                )
            analysis["potential_splits"] = list(analysis["potential_splits"])

            # Print analysis
            self._print_analysis(analysis)

            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze S3 structure: {str(e)}")
            raise

    def _is_image_file(self, key: str) -> bool:
        """Check if file is an image based on extension."""
        image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff", "tif"]
        if "." in key:
            ext = key.split(".")[-1].lower()
            return ext in image_extensions
        return False

    def _is_label_file(self, key: str) -> bool:
        """Check if file is a label file based on extension."""
        label_extensions = ["txt", "xml", "json"]
        if "." in key:
            ext = key.split(".")[-1].lower()
            return ext in label_extensions
        return False

    def _print_analysis(self, analysis: Dict):
        """Print analysis results in a readable format."""
        print("\n" + "=" * 60)
        print("S3 BUCKET ANALYSIS RESULTS")
        print("=" * 60)

        print(f"\nTotal Objects: {analysis['total_objects']}")

        print(f"\nFile Types:")
        for ext, count in sorted(analysis["file_types"].items()):
            print(f"  .{ext}: {count} files")

        print(f"\nFile Categories:")
        print(f"  Images: {len(analysis['image_files'])}")
        print(f"  Labels: {len(analysis['label_files'])}")
        print(f"  Other: {len(analysis['other_files'])}")

        print(f"\nDirectory Structure:")
        for level, dirs in analysis["directory_structure"].items():
            print(f"  {level}: {dirs}")

        # Show class information if available
        if analysis.get("class_directories"):
            print(f"\nClass Directories Found:")
            for class_name, count in sorted(analysis["class_directories"].items()):
                print(f"  {class_name}: {count} images")

            print(f"\nTotal Classes: {len(analysis['class_directories'])}")

        if analysis["potential_splits"]:
            print(
                f"\nPotential Train/Val/Test Splits Found: {analysis['potential_splits']}"
            )
        else:
            print(f"\nNo existing train/val/test splits detected")

        print("\n" + "=" * 60)

    def reorganize_data(
        self,
        source_prefix: str,
        target_prefix: str,
        split_ratios: Dict[str, float] = None,
        dry_run: bool = False,
    ) -> Dict[str, any]:
        """
        Reorganize S3 data into YOLO directory structure.

        Args:
            source_prefix: Source S3 prefix where current data is located
            target_prefix: Target S3 prefix for reorganized data
            split_ratios: Dictionary with train/val/test split ratios
            dry_run: If True, only show what would be done without executing

        Returns:
            Dictionary with reorganization results
        """
        split_ratios = split_ratios or {"train": 0.7, "val": 0.2, "test": 0.1}

        logger.info(f"Reorganizing data from '{source_prefix}' to '{target_prefix}'")
        logger.info(f"Split ratios: {split_ratios}")

        if dry_run:
            logger.info("DRY RUN MODE - No actual changes will be made")

        try:
            # Analyze current structure
            analysis = self.analyze_current_structure(source_prefix)

            # Get image and label files
            image_files = analysis["image_files"]
            label_files = analysis["label_files"]

            if not image_files:
                raise ValueError("No image files found in source prefix")

            # Match images with labels
            matched_pairs = self._match_images_with_labels(image_files, label_files)

            logger.info(f"Found {len(matched_pairs)} image-label pairs")
            logger.info(
                f"Found {len(image_files) - len(matched_pairs)} unmatched images"
            )

            # Create splits
            splits = self._create_data_splits(matched_pairs, split_ratios)

            # Reorganize files
            reorganization_plan = self._create_reorganization_plan(
                splits, source_prefix, target_prefix, analysis
            )

            if dry_run:
                self._print_reorganization_plan(reorganization_plan)
                return reorganization_plan

            # Execute reorganization
            results = self._execute_reorganization(reorganization_plan)

            # Create dataset configuration
            self._create_dataset_config(target_prefix, analysis)

            logger.info("Data reorganization completed successfully!")
            return results

        except Exception as e:
            logger.error(f"Failed to reorganize data: {str(e)}")
            raise

    def _match_images_with_labels(
        self, image_files: List[str], label_files: List[str]
    ) -> List[Tuple[str, Optional[str], Optional[str]]]:
        """
        Match image files with their corresponding label files or class information.
        For classification datasets, creates class information from directory structure.

        Args:
            image_files: List of image file keys
            label_files: List of label file keys

        Returns:
            List of tuples (image_key, label_key or None, class_name or None)
        """
        matched_pairs = []

        # Create a mapping of base names to label files
        label_map = {}
        for label_file in label_files:
            base_name = Path(label_file).stem
            label_map[base_name] = label_file

        # Match images with labels or extract class from directory
        for image_file in image_files:
            base_name = Path(image_file).stem
            label_file = label_map.get(base_name)

            # Extract class name from directory structure
            path_parts = image_file.split("/")
            class_name = path_parts[0] if len(path_parts) > 1 else None

            matched_pairs.append((image_file, label_file, class_name))

        return matched_pairs

    def _create_data_splits(
        self,
        matched_pairs: List[Tuple[str, Optional[str], Optional[str]]],
        split_ratios: Dict[str, float],
    ) -> Dict[str, List[Tuple[str, Optional[str], Optional[str]]]]:
        """
        Create train/val/test splits from matched pairs.

        Args:
            matched_pairs: List of (image, label, class_name) tuples
            split_ratios: Split ratios dictionary

        Returns:
            Dictionary with splits
        """
        # Shuffle the pairs for random splitting
        shuffled_pairs = matched_pairs.copy()
        random.shuffle(shuffled_pairs)

        total = len(shuffled_pairs)
        splits = {}
        start_idx = 0

        for split_name, ratio in split_ratios.items():
            end_idx = start_idx + int(total * ratio)
            if split_name == list(split_ratios.keys())[-1]:  # Last split gets remainder
                end_idx = total

            splits[split_name] = shuffled_pairs[start_idx:end_idx]
            start_idx = end_idx

            logger.info(f"{split_name} split: {len(splits[split_name])} pairs")

        return splits

    def _create_reorganization_plan(
        self,
        splits: Dict,
        source_prefix: str,
        target_prefix: str,
        analysis: Dict = None,
    ) -> Dict[str, List[Dict]]:
        """
        Create a plan for reorganizing files and creating YOLO labels.

        Args:
            splits: Data splits dictionary
            source_prefix: Source prefix
            target_prefix: Target prefix
            analysis: Analysis results containing class information

        Returns:
            Reorganization plan dictionary
        """
        plan = {"operations": [], "class_mapping": {}}

        # Create class mapping from analysis
        if analysis and analysis.get("class_directories"):
            class_names = sorted(analysis["class_directories"].keys())
            plan["class_mapping"] = {name: idx for idx, name in enumerate(class_names)}
            logger.info(f"Class mapping: {plan['class_mapping']}")

        for split_name, pairs in splits.items():
            for image_file, label_file, class_name in pairs:
                # Plan image file move
                image_filename = Path(image_file).name
                target_image_key = (
                    f"{target_prefix}{split_name}/images/{image_filename}"
                )

                plan["operations"].append(
                    {
                        "action": "copy",
                        "source": image_file,
                        "target": target_image_key,
                        "type": "image",
                        "split": split_name,
                        "class_name": class_name,
                    }
                )

                # Plan label file move or creation
                if label_file:
                    # Existing label file - copy it
                    label_filename = Path(label_file).name
                    target_label_key = (
                        f"{target_prefix}{split_name}/labels/{label_filename}"
                    )

                    plan["operations"].append(
                        {
                            "action": "copy",
                            "source": label_file,
                            "target": target_label_key,
                            "type": "label",
                            "split": split_name,
                            "class_name": class_name,
                        }
                    )
                elif class_name and class_name in plan["class_mapping"]:
                    # Create YOLO label from class information
                    label_filename = Path(image_file).stem + ".txt"
                    target_label_key = (
                        f"{target_prefix}{split_name}/labels/{label_filename}"
                    )

                    # For classification data, create a full-image bounding box
                    class_id = plan["class_mapping"][class_name]
                    yolo_label_content = (
                        f"{class_id} 0.5 0.5 1.0 1.0"  # Full image bbox
                    )

                    plan["operations"].append(
                        {
                            "action": "create_label",
                            "target": target_label_key,
                            "content": yolo_label_content,
                            "type": "label",
                            "split": split_name,
                            "class_name": class_name,
                            "class_id": class_id,
                        }
                    )

        return plan

    def _print_reorganization_plan(self, plan: Dict):
        """Print the reorganization plan."""
        print("\n" + "=" * 60)
        print("REORGANIZATION PLAN")
        print("=" * 60)

        operations_by_split = {}
        for op in plan["operations"]:
            split = op["split"]
            if split not in operations_by_split:
                operations_by_split[split] = {"images": 0, "labels": 0}
            operations_by_split[split][op["type"] + "s"] += 1

        for split, counts in operations_by_split.items():
            print(f"\n{split.upper()} Split:")
            print(f"  Images: {counts['images']}")
            print(f"  Labels: {counts['labels']}")

        print(f"\nTotal Operations: {len(plan['operations'])}")
        print("=" * 60)

    def _execute_reorganization(self, plan: Dict) -> Dict[str, any]:
        """
        Execute the reorganization plan.

        Args:
            plan: Reorganization plan

        Returns:
            Execution results
        """
        logger.info("Executing reorganization plan...")

        results = {
            "total_operations": len(plan["operations"]),
            "successful_operations": 0,
            "failed_operations": 0,
            "errors": [],
        }

        for i, operation in enumerate(plan["operations"]):
            try:
                target_key = operation["target"]

                if operation["action"] == "copy":
                    # Copy existing file to new location
                    source_key = operation["source"]
                    copy_source = {"Bucket": self.bucket_name, "Key": source_key}
                    self.s3_client.s3_client.copy_object(
                        CopySource=copy_source, Bucket=self.bucket_name, Key=target_key
                    )

                elif operation["action"] == "create_label":
                    # Create new YOLO label file
                    label_content = operation["content"]
                    self.s3_client.s3_client.put_object(
                        Bucket=self.bucket_name,
                        Key=target_key,
                        Body=label_content.encode("utf-8"),
                        ContentType="text/plain",
                    )

                results["successful_operations"] += 1

                if (i + 1) % 100 == 0:
                    logger.info(
                        f"Processed {i + 1}/{len(plan['operations'])} operations..."
                    )

            except Exception as e:
                action = operation.get("action", "unknown")
                source_info = (
                    f" from {operation.get('source', 'N/A')}"
                    if "source" in operation
                    else ""
                )
                error_msg = f"Failed to {action}{source_info} to {operation['target']}: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                results["failed_operations"] += 1

        logger.info(
            f"Reorganization completed: {results['successful_operations']} successful, {results['failed_operations']} failed"
        )

        return results

    def _create_dataset_config(self, target_prefix: str, analysis: Dict[str, any]):
        """
        Create a dataset configuration file in S3.

        Args:
            target_prefix: Target prefix where data was reorganized
            analysis: Analysis results containing class information
        """
        try:
            # Use actual class information from analysis
            class_directories = analysis.get("class_directories", {})
            if class_directories:
                class_names = sorted(class_directories.keys())
                num_classes = len(class_names)
            else:
                # Fallback to default classes
                num_classes = 3
                class_names = ["vehicle", "person", "building"]

            dataset_config = {
                "path": f"s3://{self.bucket_name}/{target_prefix}",
                "train": f"s3://{self.bucket_name}/{target_prefix}train/images",
                "val": f"s3://{self.bucket_name}/{target_prefix}val/images",
                "test": f"s3://{self.bucket_name}/{target_prefix}test/images",
                "nc": num_classes,
                "names": class_names,
                "created_by": "S3DataReorganizer",
                "file_types_found": analysis.get("file_types", {}),
                "class_distribution": class_directories,
            }

            # Convert to YAML format
            import yaml

            config_yaml = yaml.dump(dataset_config, default_flow_style=False)

            # Upload configuration to S3
            config_key = f"{target_prefix}dataset.yaml"
            self.s3_client.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=config_key,
                Body=config_yaml.encode("utf-8"),
                ContentType="application/x-yaml",
            )

            logger.info(
                f"Dataset configuration created at s3://{self.bucket_name}/{config_key}"
            )
            logger.info(f"Classes found: {class_names}")

        except Exception as e:
            logger.warning(f"Failed to create dataset configuration: {str(e)}")

    def cleanup_source_data(self, source_prefix: str, dry_run: bool = True):
        """
        Clean up source data after successful reorganization.

        Args:
            source_prefix: Source prefix to clean up
            dry_run: If True, only show what would be deleted
        """
        logger.warning("CLEANUP OPERATION - This will delete source data!")

        if dry_run:
            logger.info("DRY RUN MODE - No files will be deleted")

        try:
            objects = self.s3_client.list_objects(prefix=source_prefix)

            logger.info(
                f"Found {len(objects)} objects to delete with prefix '{source_prefix}'"
            )

            if dry_run:
                for obj in objects[:10]:  # Show first 10 as example
                    logger.info(f"Would delete: {obj['Key']}")
                if len(objects) > 10:
                    logger.info(f"... and {len(objects) - 10} more files")
                return

            # Delete objects in batches
            batch_size = 1000
            deleted_count = 0

            for i in range(0, len(objects), batch_size):
                batch = objects[i : i + batch_size]
                delete_objects = [{"Key": obj["Key"]} for obj in batch]

                response = self.s3_client.s3_client.delete_objects(
                    Bucket=self.bucket_name, Delete={"Objects": delete_objects}
                )

                deleted_count += len(response.get("Deleted", []))
                logger.info(f"Deleted {deleted_count}/{len(objects)} objects...")

            logger.info(f"Cleanup completed: {deleted_count} objects deleted")

        except Exception as e:
            logger.error(f"Failed to cleanup source data: {str(e)}")
            raise


def main():
    """Main function for the reorganization script."""
    parser = argparse.ArgumentParser(description="Reorganize S3 data for YOLO training")

    parser.add_argument(
        "--bucket", type=str, help="S3 bucket name (defaults to project config)"
    )
    parser.add_argument(
        "--aws-profile", type=str, default="ab", help="AWS profile to use"
    )
    parser.add_argument(
        "--source-prefix", type=str, default="", help="Source S3 prefix"
    )
    parser.add_argument(
        "--target-prefix",
        type=str,
        default="data/drone-detection/",
        help="Target S3 prefix for reorganized data",
    )

    # Split ratios
    parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Training split ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="Validation split ratio"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1, help="Test split ratio"
    )

    # Actions
    parser.add_argument(
        "--analyze-only", action="store_true", help="Only analyze current structure"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    parser.add_argument(
        "--cleanup-source",
        action="store_true",
        help="Clean up source data after reorganization",
    )

    args = parser.parse_args()

    try:
        # Get bucket name from config if not provided
        bucket_name = args.bucket
        if not bucket_name:
            config = get_config()
            bucket_name = config.get("aws", {}).get("data_bucket")
            if not bucket_name:
                raise ValueError("No bucket name provided and none found in config")

        logger.info(f"Using S3 bucket: {bucket_name}")

        # Initialize reorganizer
        reorganizer = S3DataReorganizer(bucket_name, args.aws_profile)

        if args.analyze_only:
            # Just analyze the current structure
            reorganizer.analyze_current_structure(args.source_prefix)
            return 0

        # Prepare split ratios
        split_ratios = {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        }

        # Normalize ratios to sum to 1.0
        total_ratio = sum(split_ratios.values())
        split_ratios = {k: v / total_ratio for k, v in split_ratios.items()}

        # Reorganize data
        results = reorganizer.reorganize_data(
            source_prefix=args.source_prefix,
            target_prefix=args.target_prefix,
            split_ratios=split_ratios,
            dry_run=args.dry_run,
        )

        if not args.dry_run and args.cleanup_source:
            # Clean up source data
            reorganizer.cleanup_source_data(args.source_prefix, dry_run=True)

            response = input("Are you sure you want to delete source data? (yes/no): ")
            if response.lower() == "yes":
                reorganizer.cleanup_source_data(args.source_prefix, dry_run=False)
            else:
                logger.info("Source data cleanup skipped")

        logger.info("Script completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
