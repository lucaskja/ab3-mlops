#!/usr/bin/env python3
"""
Remove Duplicate Image Files with Same Stems

This script identifies and removes duplicate image files that have the same
filename stem, keeping only one image per stem to match the labels.

Usage:
    python remove_duplicate_image_stems.py --profile ab --dry-run
    python remove_duplicate_image_stems.py --profile ab --execute
"""

import boto3
import sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse
import time
from botocore.exceptions import ClientError

class DuplicateImageStemRemover:
    def __init__(self, profile_name: str = 'ab'):
        """Initialize the remover with AWS profile."""
        self.session = boto3.Session(profile_name=profile_name)
        self.s3_client = self.session.client('s3')
        self.bucket_name = 'lucaskle-ab3-project-pv'
        
        # Image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
    def get_file_stem(self, file_path: str) -> str:
        """Get file stem (filename without extension) from S3 key."""
        return Path(file_path).stem
    
    def get_file_extension(self, file_path: str) -> str:
        """Get file extension from S3 key."""
        return Path(file_path).suffix.lower()
    
    def list_objects_with_metadata(self, prefix: str) -> list:
        """List all objects with metadata for duplicate detection."""
        objects = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        try:
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=prefix,
                PaginationConfig={'PageSize': 1000}
            )
            
            for page in page_iterator:
                if 'Contents' in page:
                    objects.extend([{
                        'Key': obj['Key'],
                        'Size': obj['Size'],
                        'LastModified': obj['LastModified'],
                        'ETag': obj['ETag']
                    } for obj in page['Contents']])
                    
        except ClientError as e:
            print(f"Error listing objects with prefix {prefix}: {e}")
            return []
            
        return objects
    
    def find_duplicate_stems_in_split(self, split: str) -> dict:
        """Find images with duplicate stems in a dataset split."""
        print(f"\n🔍 Finding duplicate image stems in {split} dataset...")
        
        # Define prefix
        images_prefix = f"datasets/yolov11_dataset_20250723_031301/{split}/images/"
        
        # Get all image objects with metadata
        print(f"📂 Scanning {images_prefix}...")
        image_objects = self.list_objects_with_metadata(images_prefix)
        
        print(f"📊 Found {len(image_objects)} image objects")
        
        # Group by stem
        stem_groups = defaultdict(list)
        
        for obj in image_objects:
            key = obj['Key']
            extension = self.get_file_extension(key)
            
            if extension in self.image_extensions:
                stem = self.get_file_stem(key)
                stem_groups[stem].append(obj)
        
        # Find stems with multiple files
        duplicate_stems = {stem: files for stem, files in stem_groups.items() if len(files) > 1}
        
        print(f"🔍 Duplicate stem analysis for {split}:")
        print(f"   Total unique stems: {len(stem_groups)}")
        print(f"   Stems with duplicates: {len(duplicate_stems)}")
        
        # Calculate total duplicate files to remove
        total_duplicates = sum(len(files) - 1 for files in duplicate_stems.values())
        print(f"   Total duplicate files to remove: {total_duplicates}")
        
        # Show samples
        if duplicate_stems:
            print(f"\n📝 Sample duplicate stems:")
            for i, (stem, files) in enumerate(list(duplicate_stems.items())[:5]):
                print(f"   {i+1}. '{stem}' ({len(files)} files):")
                for j, file_obj in enumerate(files):
                    print(f"      {j+1}. {file_obj['Key']}")
                    print(f"         Size: {file_obj['Size']}, Modified: {file_obj['LastModified']}")
        
        return {
            'split': split,
            'duplicate_stems': duplicate_stems,
            'total_duplicates': total_duplicates
        }
    
    def remove_duplicate_files(self, duplicate_stems: dict, dry_run: bool = True) -> int:
        """Remove duplicate files, keeping the first (oldest) one for each stem."""
        files_to_delete = []
        
        for stem, files in duplicate_stems.items():
            if len(files) > 1:
                # Sort by LastModified to keep the oldest
                files.sort(key=lambda x: x['LastModified'])
                
                # Mark all but the first for deletion
                for file_obj in files[1:]:
                    files_to_delete.append(file_obj['Key'])
        
        if not files_to_delete:
            return 0
        
        if dry_run:
            print(f"🔍 DRY RUN: Would delete {len(files_to_delete)} duplicate files")
            return 0
        
        # Delete files in batches
        deleted_count = 0
        batch_size = 1000
        
        for i in range(0, len(files_to_delete), batch_size):
            batch = files_to_delete[i:i + batch_size]
            
            delete_objects = {
                'Objects': [{'Key': key} for key in batch],
                'Quiet': True
            }
            
            try:
                response = self.s3_client.delete_objects(
                    Bucket=self.bucket_name,
                    Delete=delete_objects
                )
                
                if 'Deleted' in response:
                    deleted_count += len(response['Deleted'])
                
                if 'Errors' in response:
                    for error in response['Errors']:
                        print(f"❌ Error deleting {error['Key']}: {error['Message']}")
                        
            except ClientError as e:
                print(f"❌ Batch delete error: {e}")
                continue
        
        return deleted_count
    
    def remove_all_duplicate_stems(self, dry_run: bool = True) -> dict:
        """Remove duplicate image stems from both train and val splits."""
        print("🚀 Starting Duplicate Image Stem Removal")
        print(f"📍 Bucket: {self.bucket_name}")
        print(f"🔧 Mode: {'DRY RUN' if dry_run else 'EXECUTE REMOVAL'}")
        
        start_time = time.time()
        results = {'train': None, 'val': None, 'total_deleted': 0}
        
        # Process both splits
        for split in ['train', 'val']:
            duplicate_info = self.find_duplicate_stems_in_split(split)
            results[split] = duplicate_info
        
        # Calculate totals
        total_duplicates = sum(r['total_duplicates'] for r in results.values() if r)
        
        print(f"\n📋 DUPLICATE STEM SUMMARY:")
        print(f"   📦 Total duplicate files to remove: {total_duplicates}")
        
        # Remove duplicates
        if total_duplicates > 0:
            if not dry_run:
                print(f"\n🧹 REMOVING DUPLICATE FILES...")
                
                # Confirm deletion
                confirm = input(f"\n⚠️  Are you sure you want to delete {total_duplicates} duplicate files? (yes/no): ")
                if confirm.lower() != 'yes':
                    print("❌ Removal cancelled by user")
                    return results
                
                total_deleted = 0
                for split in ['train', 'val']:
                    if results[split] and results[split]['duplicate_stems']:
                        print(f"\n🧹 Removing duplicates from {split}...")
                        
                        deleted = self.remove_duplicate_files(
                            results[split]['duplicate_stems'], dry_run=False
                        )
                        total_deleted += deleted
                        print(f"   ✅ Removed {deleted} duplicate files")
                
                results['total_deleted'] = total_deleted
                print(f"\n✅ Successfully removed {total_deleted} duplicate files")
            
            else:
                print(f"\n🔍 DRY RUN: Would remove {total_duplicates} duplicate files")
                print("   Run with --execute to perform the removal")
        
        else:
            print(f"\n✅ No duplicate stems found - dataset is clean!")
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Operation completed in {elapsed_time:.2f} seconds")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Remove duplicate image files with same stems')
    parser.add_argument('--profile', default='ab', help='AWS CLI profile to use (default: ab)')
    parser.add_argument('--dry-run', action='store_true', help='Only analyze, do not delete files')
    parser.add_argument('--execute', action='store_true', help='Execute removal (delete duplicate files)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.dry_run and args.execute:
        print("❌ Error: Cannot specify both --dry-run and --execute")
        sys.exit(1)
    
    # Default to dry-run if neither specified
    dry_run = not args.execute
    
    try:
        # Initialize remover
        remover = DuplicateImageStemRemover(profile_name=args.profile)
        
        # Run duplicate removal
        results = remover.remove_all_duplicate_stems(dry_run=dry_run)
        
        # Exit with appropriate code
        total_duplicates = sum(r['total_duplicates'] for r in results.values() if r and isinstance(r, dict))
        
        if total_duplicates > 0 and dry_run:
            print(f"\n⚠️  Found {total_duplicates} duplicate files. Run with --execute to remove them.")
            sys.exit(1)
        else:
            print(f"\n✅ Duplicate removal complete!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
