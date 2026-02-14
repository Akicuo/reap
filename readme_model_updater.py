from huggingface_hub import HfApi
import os
import argparse
import re

# Argument parser for token input
parser = argparse.ArgumentParser(description="HuggingFace Model Updater")
parser.add_argument('--token', type=str, default=None, help="HuggingFace User Access Token")
args, _ = parser.parse_known_args()

# Use the provided token, or fall back to environment variable, or None
hf_token = args.token or os.getenv("HF_TOKEN")

# Initialize API with token if provided
api = HfApi(token=hf_token)

# Get your username
user_info = api.whoami()
username = user_info["name"]

# Fetch all your model repositories
my_models = list(api.list_models(author=username))

# Filter for repos ending with 'GGUF' (Case-sensitive check)
gguf_models = sorted([
    m.id for m in my_models if m.id.upper().endswith("GGUF") and "REAP" in m.id.upper()
])

# Filter for repos containing 'REAP' (Case-insensitive check) - safetensors models
reap_models = sorted([
    m.id for m in my_models if "REAP" in m.id.upper() and "GGUF" not in m.id.upper()
])

# --- Display Results ---
print(f"--- GGUF Models ({len(gguf_models)}) ---")
for model in gguf_models:
    print(model)

print(f"\n--- REAP/Safetensors Models ({len(reap_models)}) ---")
for model in reap_models:
    print(model)

# --- Update README.md ---
readme_path = "README.md"

# Generate markdown bullet lists with proper indentation
gguf_list = "\n".join([f'  - [{model}](https://huggingface.co/{model})' for model in gguf_models])
sft_list = "\n".join([f'  - [{model}](https://huggingface.co/{model})' for model in reap_models])

# Read the README
try:
    with open(readme_path, 'r', encoding='utf-8') as f:
        readme_content = f.read()
except FileNotFoundError:
    print(f"Error: {readme_path} not found!")
    exit(1)

# Replace content between HTML comments with markdown lists
# Pattern for Safetensors
sft_pattern = r'<!-- BEGIN-SFTNS -->.*?<!-- END-SFTNS -->'
sft_replacement = f'<!-- BEGIN-SFTNS -->\n{sft_list}\n  <!-- END-SFTNS -->'

# Pattern for GGUF
gguf_pattern = r'<!-- BEGIN-GGUF -->.*?<!-- END-GGUF -->'
gguf_replacement = f'<!-- BEGIN-GGUF -->\n{gguf_list}\n  <!-- END-GGUF -->'

# Update content with DOTALL flag to match across newlines
new_content = readme_content
new_content = re.sub(sft_pattern, sft_replacement, new_content, flags=re.DOTALL)
new_content = re.sub(gguf_pattern, gguf_replacement, new_content, flags=re.DOTALL)

# Write back to README
with open(readme_path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f"\n✓ Updated {readme_path} with model lists")
print(f"  - {len(gguf_models)} GGUF models")
print(f"  - {len(reap_models)} REAP/Safetensors models")
