# download_leffa_windows.py
import os
from huggingface_hub import snapshot_download # Ensure huggingface_hub is installed

# 1. Define repository details
repo_id = "franciszzj/Leffa"

# 2. Determine the target 'ckpts' directory
# This will be a 'ckpts' folder in the same directory where the script is run.
current_script_execution_dir = os.getcwd()
target_ckpts_dir = os.path.join(current_script_execution_dir, "ckpts")

# Create the target 'ckpts' directory if it doesn't exist
if not os.path.exists(target_ckpts_dir):
    print(f"Directory '{target_ckpts_dir}' not found. Creating it...")
    os.makedirs(target_ckpts_dir, exist_ok=True)
else:
    print(f"Using existing directory: '{target_ckpts_dir}'")

# Files/patterns to always ignore during a full download.
ignore_list_for_full_download = [
    "pose_transfer.pth",
    "*.git/*",
    "virtual_tryon_dc.pth",
    "virtual_tryon.pth",
    # You can add other patterns like "*.md", ".gitattributes" if you don't want them
]

# 3. Define essential model files and their expected relative paths (INSIDE the ckpts_dir)
essential_model_files = {
    "densepose_config": os.path.join("densepose", "densepose_rcnn_R_50_FPN_s1x.yaml"),
    "densepose_weights": os.path.join("densepose", "model_final_162be9.pkl"),
    "humanparsing_atr": os.path.join("humanparsing", "parsing_atr.onnx"),
    "humanparsing_lip": os.path.join("humanparsing", "parsing_lip.onnx"),
    "openpose_body": os.path.join("openpose", "body_pose_model.pth"),
    "sd_inpainting_folder": "stable-diffusion-inpainting", # This is a subfolder within ckpts_dir
    # This will be at the root of ckpts_dir if downloaded
    # This will be at the root of ckpts_dir if downloaded
}

def check_essential_files_exist(base_path_for_check, files_to_check):
    """Checks if all essential files exist relative to the base_path_for_check."""
    all_exist = True
    missing_files_details = []
    print(f"\nChecking for essential model files inside '{base_path_for_check}'...")
    for key, relative_path_in_ckpts in files_to_check.items():
        full_path_to_check = os.path.join(base_path_for_check, relative_path_in_ckpts)
        
        # For 'sd_inpainting_folder', we expect it to be a directory containing specific files.
        # A more robust check would be to see if 'sd_inpainting_folder/model_index.json' exists, for example.
        if key == "sd_inpainting_folder": 
            # Check if the folder itself exists
            if os.path.isdir(full_path_to_check):
                print(f"  [FOUND] Directory: {relative_path_in_ckpts}")
                # Optionally, check for a key file within it, e.g., model_index.json
                # if not os.path.isfile(os.path.join(full_path_to_check, "model_index.json")):
                #     print(f"    [WARNING] Directory '{relative_path_in_ckpts}' exists, but a key file (e.g., model_index.json) is missing.")
                #     # Decide if this constitutes "missing" for your needs
            else:
                print(f"  [MISSING] Directory: {relative_path_in_ckpts}")
                missing_files_details.append(relative_path_in_ckpts)
                all_exist = False
        else: # Check for file
            if os.path.isfile(full_path_to_check):
                print(f"  [FOUND] File: {relative_path_in_ckpts}")
            else:
                print(f"  [MISSING] File: {relative_path_in_ckpts}")
                missing_files_details.append(relative_path_in_ckpts)
                all_exist = False
    if missing_files_details:
        print(f"\nThe following essential files/folders are missing from '{base_path_for_check}': {missing_files_details}")
    return all_exist

# 4. Perform the check and conditional download
if check_essential_files_exist(target_ckpts_dir, essential_model_files):
    print("\nAll essential model files already exist in the target directory.")
    print(f"Target directory: '{target_ckpts_dir}'")
    print("Skipping repository download.")
else:
    print("\nSome essential model files are missing from the target directory.")
    print(f"Target directory for download: '{target_ckpts_dir}'")
    print(f"--- IMPORTANT ---")
    print(f"Files and folders from '{repo_id}' will be downloaded into: \n{target_ckpts_dir}")
    print(f"This may overwrite existing files in this target directory if they have the same names.")
    print(f"Ignoring the following patterns during download: {ignore_list_for_full_download}")
    print(f"\nAutomatically proceeding with download of repository '{repo_id}' to '{target_ckpts_dir}'...")
    
    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=target_ckpts_dir, # Download into the 'ckpts' folder
            ignore_patterns=ignore_list_for_full_download,
            local_dir_use_symlinks=False,
            # token=None, # Use your HF token string or True if repo is private
            # resume_download=True, # Useful for large downloads
            # force_download=False # Set to True to re-download even if files exist (not recommended here)
        )
        print(f"\nDownload complete! Files are in '{os.path.abspath(downloaded_path)}'.")
        # Re-verify after download
        print("\nRe-verifying essential files after download...")
        check_essential_files_exist(target_ckpts_dir, essential_model_files)
    except Exception as e:
        print(f"\nAn error occurred during download: {e}")

# 5. List top-level contents of the target 'ckpts' directory
print(f"\nTop-level contents of '{target_ckpts_dir}':")
try:
    if os.path.exists(target_ckpts_dir):
        for item in sorted(os.listdir(target_ckpts_dir)):
            item_path = os.path.join(target_ckpts_dir, item)
            item_type = "Folder" if os.path.isdir(item_path) else "File"
            print(f"  - {item} ({item_type})")
    else:
        print(f"Target directory '{target_ckpts_dir}' does not exist (this shouldn't happen if creation was successful).")
except Exception as e:
    print(f"Could not list contents of '{target_ckpts_dir}': {e}")

# 6. Final check for the explicitly excluded file within the 'ckpts' directory
excluded_file_path_in_ckpts = os.path.join(target_ckpts_dir, "pose_transfer.pth")
if not os.path.exists(excluded_file_path_in_ckpts):
    print(f"\n  [CORRECTLY EXCLUDED or NOT PRESENT in '{target_ckpts_dir}'] pose_transfer.pth")
else:
    print(f"\n  [NOTE] pose_transfer.pth exists in '{target_ckpts_dir}' (it might have been downloaded previously if not ignored, or if it's part of another model).")