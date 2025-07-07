import os
from llama_cpp import Llama


# Cache the model instance to avoid reloading
_llm_instance = None

def load_llama_model(model_path: str, n_ctx: int = 2048, n_threads: int = None) -> Llama:
    """
    Loads a LLaMA model from the given path. Uses a cached instance if already loaded.

    Args:
        model_path (str): Absolute or relative path to the LLaMA model file (.gguf)
        n_ctx (int): Number of context tokens
        n_threads (int): Number of CPU threads to use (optional)

    Returns:
        Llama: Llama model instance
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    print(f"🧠 Loading LLaMA model from: {model_path}")
    _llm_instance = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads or os.cpu_count(),
        verbose=False
    )
    return _llm_instance


def generate_stock_comment(llm: Llama, prompt: str, max_tokens: int = 512) -> str:
    """
    Generates a stock analysis comment using the LLaMA model.

    Args:
        llm (Llama): LLaMA model instance
        prompt (str): Prompt string to feed to the model
        max_tokens (int): Maximum tokens to generate

    Returns:
        str: Generated model output text
    """
    print(f"📨 Prompting LLaMA: {prompt}")
    response = llm(prompt, max_tokens=max_tokens, stop=["\n"])

    if isinstance(response, dict):
        return response["choices"][0]["text"].strip()
    else:
        return str(response).strip()
