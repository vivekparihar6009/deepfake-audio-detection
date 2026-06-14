import os
import argparse
import zipfile

def download_and_extract(data_dir):
    # Set Kaggle config path to default location or check environment variables
    # Since Kaggle requires kaggle.json to authenticate, it looks at ~/.kaggle/
    # If the user has KAGGLE_USERNAME and KAGGLE_KEY, the API handles them automatically.
    
    # Import kaggle inside to ensure environment variables or files are set before execution
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        print("Error importing Kaggle API. Make sure you have 'kaggle' package installed.")
        raise e
        
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print("\n" + "="*80)
        print("KAGGLE AUTHENTICATION ERROR")
        print("Could not authenticate with Kaggle API.")
        print("Please ensure you have placed your 'kaggle.json' API token in:")
        print("  C:\\Users\\vivek parihar\\.kaggle\\kaggle.json")
        print("Or set the environment variables KAGGLE_USERNAME and KAGGLE_KEY.")
        print("="*80 + "\n")
        raise e
        
    dataset = "mohammedabdeldayem/the-fake-or-real-dataset"
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "the-fake-or-real-dataset.zip")
    
    print(f"Downloading dataset '{dataset}' to '{zip_path}'...")
    # This downloads the zip file without extracting all
    api.dataset_download_files(dataset, path=data_dir, unzip=False, quiet=False)
    
    print("Download complete. Starting selective extraction of 'for-norm' folder...")
    
    # The extraction folder will be direct
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        all_files = zip_ref.namelist()
        
        # Filter only files in the 'for-norm' sub-folder
        norm_files = [f for f in all_files if "for-norm" in f]
        
        if not norm_files:
            print("Warning: No files containing 'for-norm' found in the zip. Here are some directories:")
            # Print unique top level folders in zip
            top_level = set([f.split('/')[0] for f in all_files if '/' in f])
            print("Top level directories:", top_level)
            print("Extracting all files instead...")
            norm_files = all_files
            
        print(f"Found {len(norm_files)} files to extract. Extracting to '{data_dir}'...")
        
        # Extract selective files
        for i, file in enumerate(norm_files):
            zip_ref.extract(file, data_dir)
            if (i + 1) % 5000 == 0:
                print(f"Extracted {i + 1}/{len(norm_files)} files...")
                
    print("Selective extraction finished successfully!")
    
    # Delete the large zip file
    if os.path.exists(zip_path):
        print("Cleaning up zip file...")
        os.remove(zip_path)
        print("Zip file deleted successfully to save space.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract Fake-or-Real Audio Deepfake dataset.")
    parser.add_argument("--data_dir", type=str, default="D:\\kaggle-data", help="Directory to store the dataset.")
    args = parser.parse_args()
    
    download_and_extract(args.data_dir)
