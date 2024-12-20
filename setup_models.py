import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables from .env
load_dotenv()


def download_model(model_name, save_dir="models"):
    # Set Hugging Face API token
    api_token = os.getenv("HUGGINGFACE_API_TOKEN")
    if api_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = api_token

    # Load and cache the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=save_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=save_dir)
    print(f"Model and tokenizer downloaded to {save_dir}")


if __name__ == "__main__":
    download_model("meta-llama/Llama-3.2-3B")
