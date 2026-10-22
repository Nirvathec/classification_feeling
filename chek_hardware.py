"""
Diagnostic du matériel disponible pour l'entraînement (CPU, RAM, GPU NVIDIA, CUDA, PyTorch).

Aucune dépendance obligatoire : PyTorch est testé s'il est installé, ignoré sinon.
Usage : python check_hardware.py
"""

import os
import platform
import re
import subprocess
import sys


def run(cmd):
    """Exécute une commande et renvoie sa sortie, ou None si elle échoue ou n'existe pas."""
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return res.stdout.strip() if res.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def ram_total_gb():
    """RAM totale en Go, via psutil si présent, sinon via l'API du système."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except ImportError:
        pass
    if platform.system() == "Windows":
        import ctypes

        class MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemStatus()
        status.dwLength = ctypes.sizeof(MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullTotalPhys / 1e9
    if os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo") as f:
            return int(f.readline().split()[1]) * 1024 / 1e9
    return None


def section(title):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------- Système
section("Système")
print(f"OS         : {platform.system()} {platform.release()}")
print(f"Python     : {sys.version.split()[0]} ({sys.executable})")
print(f"CPU        : {platform.processor() or 'non détecté'} - {os.cpu_count()} threads logiques")
ram = ram_total_gb()
print(f"RAM        : {f'{ram:.1f} Go' if ram else 'non détectée'}")

# ---------------------------------------------------------------- GPU / pilote
section("GPU NVIDIA et pilote")
smi_full = run(["nvidia-smi"])
driver_cuda = None
if smi_full is None:
    print("nvidia-smi introuvable : aucun pilote NVIDIA détecté (ou GPU non NVIDIA).")
else:
    query = run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader",
    ])
    if query is None:  # anciens pilotes : le champ compute_cap n'existe pas
        query = run([
            "nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"
        ])
    for i, line in enumerate(query.splitlines() if query else []):
        parts = [p.strip() for p in line.split(",")]
        print(f"GPU {i}      : {parts[0]}")
        print(f"  VRAM       : {parts[1]}")
        print(f"  Pilote     : {parts[2]}")
        if len(parts) > 3:
            print(f"  Compute cap: {parts[3]}")
    match = re.search(r"CUDA Version:\s*([\d.]+)", smi_full)
    if match:
        driver_cuda = match.group(1)
        print(f"CUDA maximale prise en charge par le pilote : {driver_cuda}")

# ---------------------------------------------------------------- CUDA Toolkit
section("CUDA Toolkit (nvcc)")
nvcc = run(["nvcc", "--version"])
if nvcc:
    match = re.search(r"release\s+([\d.]+)", nvcc)
    print(f"nvcc détecté : CUDA {match.group(1) if match else '?'}")
else:
    print("nvcc non détecté. Ce n'est pas bloquant : les versions CUDA de PyTorch")
    print("embarquent leur propre runtime CUDA, seul le pilote est nécessaire.")

# ---------------------------------------------------------------- PyTorch
section("PyTorch")
torch_ok = False
try:
    import torch

    print(f"Version PyTorch : {torch.__version__}")
    print(f"Compilé avec CUDA : {torch.version.cuda or 'NON (version CPU uniquement)'}")
    torch_ok = torch.cuda.is_available()
    print(f"torch.cuda.is_available() : {torch_ok}")
    if torch_ok:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i} vu par PyTorch : {props.name}")
            print(f"  VRAM         : {props.total_memory / 1e9:.1f} Go")
            print(f"  Compute cap. : {props.major}.{props.minor}")
        print(f"bfloat16 supporté : {torch.cuda.is_bf16_supported()}")
except ImportError:
    torch = None
    print("PyTorch n'est pas installé dans cet environnement.")

# ---------------------------------------------------------------- Verdict
section("Verdict")
if torch_ok:
    print("[OK] GPU utilisable avec PyTorch. Entraînement sur GPU possible.")
elif smi_full is None:
    print("[!!] Aucun pilote NVIDIA détecté. Installer ou mettre à jour le pilote NVIDIA.")
elif torch is None:
    print("[!!] Pilote OK mais PyTorch absent : installer une version PyTorch avec CUDA.")
elif torch.version.cuda is None:
    print("[!!] PyTorch installé en version CPU uniquement : le GPU existe mais n'est pas utilisé.")
    print("     Réinstaller PyTorch avec CUDA (commande sur pytorch.org, rubrique 'Get Started').")
    print("     Attention : un requirements.txt contenant l'index 'whl/cpu' (cas du déploiement")
    print("     Render) réinstalle la version CPU. Garder deux fichiers de dépendances distincts.")
else:
    print("[!!] PyTorch est compilé avec CUDA mais ne voit pas le GPU.")
    print("     Pistes : pilote trop ancien pour la version CUDA de PyTorch, ou environnement")
    print("     différent de celui attendu (vérifier le chemin Python affiché plus haut).")