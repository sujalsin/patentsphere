"""Verify that required Ollama models are available."""
import httpx
import sys
from config import settings


def check_ollama_connection():
    """Check if Ollama is running."""
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except Exception as e:
        return False, f"Error connecting to Ollama: {e}"


def verify_models():
    """Verify all required models are available."""
    print("🔍 Checking Ollama connection...")
    connected, data = check_ollama_connection()
    
    if not connected:
        print(f"❌ Cannot connect to Ollama at {settings.ollama_base_url}")
        print("   Please ensure Ollama is running.")
        return False
    
    print(f"✅ Connected to Ollama at {settings.ollama_base_url}\n")
    
    # Get list of available models
    if isinstance(data, dict):
        available_models = [model.get("name", "") for model in data.get("models", [])]
    else:
        print("❌ Could not retrieve model list")
        return False
    
    # Required models
    required_models = [
        settings.ollama_router_model,
        settings.ollama_extractor_model,
        settings.ollama_synthesizer_model,
        settings.ollama_critic_model,
    ]
    
    print("📋 Checking required models:\n")
    all_present = True
    missing_models = []
    
    for model in required_models:
        # Check if model exists (exact match or starts with)
        found = False
        for available in available_models:
            if model in available or available.startswith(model.split(":")[0]):
                found = True
                print(f"  ✅ {model}")
                break
        
        if not found:
            print(f"  ❌ {model} - MISSING")
            missing_models.append(model)
            all_present = False
    
    print()
    
    if all_present:
        print("✅ All required models are available!")
        return True
    else:
        print("❌ Some models are missing. Please install them using:")
        print()
        for model in missing_models:
            print(f"   ollama pull {model}")
        print()
        return False


if __name__ == "__main__":
    success = verify_models()
    sys.exit(0 if success else 1)


