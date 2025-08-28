#!/usr/bin/env python3
"""
Alternative installation script for Lighthouse HealthConnect
Run this if requirements.txt installation fails
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False
    return True

def main():
    # Core packages first
    core_packages = [
        "python-dotenv==1.0.1",
        "requests==2.31.0",
        "numpy==1.26.4",
        "tqdm==4.66.4",
    ]
    
    # LangChain packages
    langchain_packages = [
        "langchain-core==0.1.52",
        "langchain==0.1.20",
        "langchain-community==0.0.38",
        "langchain-openai==0.1.7",
        "langchain-pinecone==0.1.0",
        "langchain-huggingface==0.0.3",
    ]

    # ML/AI packages
    ml_packages = [
        "sentence-transformers==2.7.0",
        "scikit-learn==1.4.2",
        "huggingface_hub==0.23.0",
    ]
    
    # Vector database
    vector_packages = [
        "pinecone-client>=3.0.0",
    ]
    
    # Document processing
    doc_packages = [
        "pypdf==4.2.0",
        "beautifulsoup4==4.12.3",
        "unstructured==0.13.7",
        "langdetect==1.0.9",
        "markdown==3.6",
    ]
    
    # Audio packages
    audio_packages = [
        "SpeechRecognition==3.10.0",
        "gTTS==2.4.0",
        "pydub==0.25.1",
    ]
    
    # Streamlit packages
    ui_packages = [
        "streamlit>=1.30.0",
        "streamlit-audiorec>=0.1.0",
    ]
    
    # Optional packages (install last)
    optional_packages = [
        "openai-whisper",
    ]
    
    all_package_groups = [
        ("Core packages", core_packages),
        ("LangChain packages", langchain_packages),
        ("ML packages", ml_packages),
        ("Vector database", vector_packages),
        ("Document processing", doc_packages),
        ("Audio packages", audio_packages),
        ("UI packages", ui_packages),
        ("Optional packages", optional_packages),
    ]
    
    print("🚀 Starting package installation for Lighthouse HealthConnect")
    print("=" * 60)
    
    failed_packages = []
    
    for group_name, packages in all_package_groups:
        print(f"\n📦 Installing {group_name}...")
        print("-" * 40)
        
        for package in packages:
            if not install_package(package):
                failed_packages.append(package)
    
    # Try to install torch separately (it's often problematic)
    print(f"\n🔥 Installing PyTorch...")
    print("-" * 40)
    torch_cmd = [
        sys.executable, "-m", "pip", "install", 
        "torch==2.2.2", 
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ]
    
    try:
        subprocess.check_call(torch_cmd)
        print("✅ Successfully installed PyTorch")
    except subprocess.CalledProcessError:
        print("❌ Failed to install PyTorch with CPU index, trying default...")
        if not install_package("torch==2.2.2"):
            failed_packages.append("torch")
    
    print("\n" + "=" * 60)
    print("🏁 Installation Summary")
    print("=" * 60)
    
    if failed_packages:
        print(f"❌ Failed to install {len(failed_packages)} packages:")
        for package in failed_packages:
            print(f"  - {package}")
        print("\nYou can try installing these manually or check for compatibility issues.")
    else:
        print("✅ All packages installed successfully!")
    
    print(f"\n💡 Next steps:")
    print("1. Set up your .env file with API keys")
    print("2. Create a 'data' directory and add your knowledge base ZIP file")
    print("3. Run: python -m streamlit run app.py")

if __name__ == "__main__":
    main()