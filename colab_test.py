# colab_test.py
import sys

def run_diagnostic():
    print("🚀 [SUCCESS] VS Code code successfully pulled into Colab!")
    
    # Check if we can see our persistent libraries folder
    drive_package_path = '/content/drive/MyDrive/colab_packages'
    if drive_package_path in sys.path:
        print("📁 [SUCCESS] Google Drive package path is successfully injected.")
    else:
        print("⚠️ [WARNING] Google Drive package path not found in sys.path.")

    # Try importing a library to ensure the Drive path works
    try:
        # Swap 'openai' with whatever library you actually installed there
        import openai 
        print(f"📦 [SUCCESS] Successfully imported 'openai' from permanent Drive storage!")
    except ImportError:
        print("ℹ️ [INFO] Drive path is set, but no packages are installed in it yet.")

if __name__ == "__main__":
    run_diagnostic()