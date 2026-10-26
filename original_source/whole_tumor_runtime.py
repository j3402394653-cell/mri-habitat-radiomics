"""Locate a working radiomics interpreter before importing third-party packages."""
import os
from pathlib import Path
import subprocess
import sys

MODULES = ('numpy', 'pandas', 'nibabel', 'SimpleITK', 'radiomics')


def ensure_runtime(modules=MODULES):
    import importlib
    try:
        for module in modules:
            importlib.import_module(module)
        return
    except (ImportError, OSError) as error:
        reason = str(error)
    candidates = [os.environ.get('HABITAT_PYTHON'),
                  str(Path(__file__).parent / '.venv' / 'Scripts' / 'python.exe'),
                  r'D:\Anaconda\envs\habitat_radiomics\python.exe']
    probe = 'import ' + ','.join(modules)
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        if Path(candidate).resolve() == Path(sys.executable).resolve():
            continue
        try:
            check = subprocess.run([candidate, '-c', probe], capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if check.returncode == 0:
            print(f'Python dependencies unavailable in {sys.executable}: {reason}', flush=True)
            print(f'Using radiomics environment: {candidate}', flush=True)
            raise SystemExit(subprocess.call([candidate, '-X', 'utf8', *sys.argv]))
    raise SystemExit(
        f'Radiomics dependencies unavailable in {sys.executable}: {reason}\n'
        'Use your habitat_radiomics Python, or set HABITAT_PYTHON to its python.exe.\n'
        'Required packages: numpy pandas nibabel SimpleITK pyradiomics.'
    )
