import subprocess
import os
import sys

LOCK_FILE = ".git/index.lock"

def remove_lock():
    if os.path.exists(LOCK_FILE):
        try:
            os.remove(LOCK_FILE)
        except:
            pass

def get_all_files(root_dir):
    file_list = []
    ignored_dirs = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'env', '.idea', '.vscode', 'dist', 'build', 'coverage'}
    
    for root, dirs, files in os.walk(root_dir):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, root_dir)
            file_list.append(rel_path)
    return file_list

print("Gathering file list...")
all_files = get_all_files(".")
print(f"Found {len(all_files)} files.")

for i, file_path in enumerate(all_files):
    if i % 100 == 0:
        print(f"Processing {i}/{len(all_files)}...")
        
    remove_lock()
    
    # Try to add the file
    result = subprocess.run(['git', 'add', file_path], capture_output=True, text=True)
    
    if result.returncode != 0:
        # Check if it's just ignored
        if "ignored by one of your .gitignore files" in result.stderr:
            continue
            
        print(f"Failed to add {file_path}. Return code: {result.returncode}")
        # print(f"Stderr: {result.stderr}")
        
        # If it crashed (negative return code) or gave short read error
        if result.returncode < 0 or "short read" in result.stderr or "unable to index" in result.stderr:
            print(f"Deleting corrupted file: {file_path}")
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

print("Finished processing all files.")

# Try final add
remove_lock()
subprocess.run(['git', 'add', '-A'], check=True)
print("Final git add -A successful.")
