import os
import platform
import urllib.request
import tarfile
import zipfile
import shutil

VERSION = "v0.4.10"
BASE_URL = f"https://github.com/DeusData/codebase-memory-mcp/releases/download/{VERSION}/"

def get_platform_info():
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        os_name = "darwin"
    elif system == "linux":
        os_name = "linux"
    elif system == "windows":
        os_name = "windows"
    else:
        raise RuntimeError(f"Unsupported operating system: {system}")

    if machine in ["x86_64", "amd64"]:
        arch = "amd64"
    elif machine in ["aarch64", "arm64"]:
        arch = "arm64"
    else:
        raise RuntimeError(f"Unsupported architecture: {machine}")

    return os_name, arch

def main():
    try:
        os_name, arch = get_platform_info()
    except Exception as e:
        print(f"Error checking platform: {e}")
        return

    extension = "zip" if os_name == "windows" else "tar.gz"
    filename = f"codebase-memory-mcp-{os_name}-{arch}.{extension}"
    download_url = BASE_URL + filename

    bin_dir = os.path.join(os.path.dirname(__file__), "bin")
    os.makedirs(bin_dir, exist_ok=True)

    download_path = os.path.join(bin_dir, filename)

    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(download_url, download_path)
    except Exception as e:
        print(f"Error downloading the file from {download_url}: {e}")
        return

    print("Extracting...")
    try:
        if extension == "zip":
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                zip_ref.extractall(bin_dir)
        else:
            with tarfile.open(download_path, "r:gz") as tar_ref:
                tar_ref.extractall(bin_dir)
                
        # Handle executable permissions for Unix-like systems
        if os_name != "windows":
            extracted_bin = os.path.join(bin_dir, "codebase-memory-mcp")
            if os.path.exists(extracted_bin):
                os.chmod(extracted_bin, 0o755)
    except Exception as e:
        print(f"Error extracting the file: {e}")
        return
    finally:
        if os.path.exists(download_path):
            os.remove(download_path)

    print(f"Successfully installed codebase-memory-mcp into {bin_dir}")

if __name__ == "__main__":
    main()
