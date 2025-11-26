import subprocess
import os
import sys
import re

max_iterations = 500
for i in range(max_iterations):
    print(f"Iteration {i+1}...")
    
    result = subprocess.run(['git', 'commit', '-m', 'Fix corrupted repository: removed damaged files and duplicates'], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Success! git commit completed.")
        sys.exit(0)
        
    stderr = result.stderr
    print(f"Git commit failed. Return code: {result.returncode}")
    
    # Look for the specific error pattern
    # error: invalid object 100644 d44f3e5ae2e3beec3aeca8b7ce35a91eeeb8d99a for '.serena/memories/autopilot_session_20251011.md'
    
    match = re.search(r"error: invalid object \d+ [a-f0-9]+ for '(.+)'", stderr)
    
    if match:
        file_path = match.group(1).strip()
        print(f"Found invalid object for file: {file_path}")
        
        # Remove from index
        cmd = ['git', 'rm', '--cached', file_path]
        print(f"Running: {' '.join(cmd)}")
        rm_result = subprocess.run(cmd, capture_output=True, text=True)
        
        if rm_result.returncode != 0:
            print(f"Failed to remove {file_path} from index: {rm_result.stderr}")
            # Try forcing if needed, or maybe it's already gone?
            # If it fails, we might be stuck.
            
    else:
        print("Could not identify an invalid object from the error output.")
        print("Full stderr:")
        print(stderr)
        sys.exit(1)

print("Max iterations reached without success.")
sys.exit(1)
