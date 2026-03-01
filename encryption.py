"""
encryption.py - AES Encryption Module using Fernet (AES-128 CBC)
Handles all encryption/decryption operations for EEG files and reports.
"""

import os
import json
from cryptography.fernet import Fernet

# Key file path
KEY_FILE = "secret.key"
ENCRYPTED_DIR = "encrypted_data"

def generate_key():
    """Generate and save a new Fernet (AES) encryption key."""
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    print("[+] Encryption key generated and saved to secret.key")
    return key

def load_key():
    """Load the encryption key from file, generate if not exists."""
    if not os.path.exists(KEY_FILE):
        return generate_key()
    with open(KEY_FILE, "rb") as f:
        return f.read()

def get_fernet():
    """Return a Fernet cipher instance using the loaded key."""
    key = load_key()
    return Fernet(key)

def encrypt_file(file_path: str) -> str:
    """
    Encrypt a file and store it in the encrypted_data directory.
    Returns the path to the encrypted file.
    """
    fernet = get_fernet()
    os.makedirs(ENCRYPTED_DIR, exist_ok=True)

    with open(file_path, "rb") as f:
        original_data = f.read()

    encrypted_data = fernet.encrypt(original_data)

    # Save encrypted file with .enc extension
    filename = os.path.basename(file_path)
    enc_filename = filename + ".enc"
    enc_path = os.path.join(ENCRYPTED_DIR, enc_filename)

    with open(enc_path, "wb") as f:
        f.write(encrypted_data)

    print(f"[+] File encrypted: {enc_path}")
    return enc_path

def decrypt_file(enc_path: str, output_path: str) -> str:
    """
    Decrypt an encrypted file and write to output_path.
    Returns path to decrypted file.
    """
    fernet = get_fernet()

    with open(enc_path, "rb") as f:
        encrypted_data = f.read()

    decrypted_data = fernet.decrypt(encrypted_data)

    with open(output_path, "wb") as f:
        f.write(decrypted_data)

    print(f"[+] File decrypted: {output_path}")
    return output_path

def encrypt_report(report_dict: dict) -> bytes:
    """Encrypt a prediction report (dict) and return encrypted bytes."""
    fernet = get_fernet()
    json_bytes = json.dumps(report_dict).encode("utf-8")
    return fernet.encrypt(json_bytes)

def decrypt_report(encrypted_bytes: bytes) -> dict:
    """Decrypt an encrypted report and return the dict."""
    fernet = get_fernet()
    json_bytes = fernet.decrypt(encrypted_bytes)
    return json.loads(json_bytes.decode("utf-8"))

def encrypt_text(text: str) -> str:
    """Encrypt a string and return base64 encoded string."""
    fernet = get_fernet()
    return fernet.encrypt(text.encode()).decode()

def decrypt_text(token: str) -> str:
    """Decrypt a base64 encoded encrypted string."""
    fernet = get_fernet()
    return fernet.decrypt(token.encode()).decode()
