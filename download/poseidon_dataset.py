from huggingface_hub import snapshot_download
import subprocess
from pathlib import Path

datasets = [
    "NS-PwC", "NS-BB", "CE-RPUI", "CE-RM", "Wave-Layer", "Wave-Gauss"
]

for name in datasets:
    repo_id = f"camlab-ethz/{name}"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=f"poseidon/camlab-ethz/down_streams/{name}",
        local_dir_use_symlinks=False
    )


for dataset in datasets:
    assemble_script = f"poseidon/camlab-ethz/down_streams/{dataset}/assemble_data.py" 
    input_dir = f"poseidon/camlab-ethz/down_streams/{dataset}"
    output_file = f"{input_dir}.nc"

    if not Path(input_dir).exists():
        print(f"[Skipped] Input directory does not exist: {input_dir}")
        continue

    cmd = [
        "python",
        assemble_script,
        "--input_dir", input_dir,
        "--output_file", output_file
    ]

    print(f"[Processing] {dataset}")
    try:
        subprocess.run(cmd, check=True)
        print(f"[Completed] {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"[Failed] {dataset}: {e}")