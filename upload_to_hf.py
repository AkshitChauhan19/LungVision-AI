import os
import sys
from huggingface_hub import HfApi

def deploy(token: str, space_name: str = "lungvision-ai"):
    print("[*] Authenticating with Hugging Face...")
    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/{space_name}"
    
    print(f"[*] Creating permanent 24/7 Space: {repo_id}...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True
    )
    
    deploy_folder = os.path.join(os.path.dirname(__file__), "deploy_hf")
    print(f"[*] Uploading LungVision AI application and model checkpoints from {deploy_folder}...")
    api.upload_folder(
        folder_path=deploy_folder,
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy LungVision AI 24/7 Platform"
    )
    
    public_url = f"https://{username.lower()}-{space_name.lower()}.hf.space"
    hub_url = f"https://huggingface.co/spaces/{repo_id}"
    
    print("\n" + "=" * 65)
    print("  PERMANENT 24/7 CLOUD DEPLOYMENT INITIALIZED!")
    print(f"  Hugging Face Space:  {hub_url}")
    print(f"  Direct Public URL:   {public_url}")
    print("=" * 65 + "\n")
    return public_url, hub_url

if __name__ == "__main__":
    token = os.environ.get("HF_TOKEN") or (sys.argv[1] if len(sys.argv) >= 2 else None)
    if not token:
        print("Usage: set HF_TOKEN env var or run: python upload_to_hf.py <HF_WRITE_TOKEN>")
        sys.exit(1)
    deploy(token)
