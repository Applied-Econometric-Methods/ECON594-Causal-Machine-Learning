"""
Check that your course environment is set up correctly.

Run it from the folder you cloned, with the cml environment active:

    conda activate cml
    python check_setup.py

Every line should read ok. Anything that reads MISSING names the package that
did not install, so you can rerun just that piece.
"""

import importlib
import platform
import sys

# (import name, name you install it under, what it is for)
PACKAGES = [
    ("numpy",        "numpy",              "arrays and numerical work"),
    ("pandas",       "pandas",             "data frames"),
    ("scipy",        "scipy",              "statistics and optimization"),
    ("matplotlib",   "matplotlib",         "plots"),
    ("seaborn",      "seaborn",            "nicer plots"),
    ("sklearn",      "scikit-learn",       "the machine learning methods"),
    ("statsmodels",  "statsmodels",        "regression with standard errors"),
    ("linearmodels", "linearmodels",       "panel and instrumental variables"),
    ("networkx",     "networkx",           "graphs, used for the causal diagrams"),
    ("imageio_ffmpeg", "imageio-ffmpeg",   "the encoder for lecture animations"),
    ("fklearn",      "fklearn",            "gain curves in the regularization lecture"),
    ("torch",        "torch",              "the neural network lectures"),
    ("graphviz",     "python-graphviz",    "drawing the causal diagrams"),
    ("jupyterlab",   "jupyterlab",         "the notebook interface"),
    ("ipykernel",    "ipykernel",          "what actually runs your Python"),
    ("pywrangling",  "pywrangling",        "the course toolkit"),
]

OPTIONAL = {"fklearn", "graphviz", "torch"}

# Why each optional package is survivable, so the message fits what is actually missing.
SURVIVABLE = {
    "fklearn": "only used for the gain curves in the regularization lecture",
    "graphviz": "the lectures fall back to plainer diagrams drawn with matplotlib",
    "torch": "only needed once the neural network lectures start",
}


def graphviz_really_works():
    """Importing graphviz is not enough. It also needs the dot program itself,
    and the classic failure is having one without the other."""
    try:
        import graphviz
    except Exception:
        return False, "not installed"
    try:
        graphviz.Digraph().pipe(format="svg")
        return True, ""
    except Exception:
        return False, "installed, but the dot drawing program will not run"


def main():
    print("Python  ", sys.version.split()[0], "on", platform.system())
    print("Running ", sys.executable)
    print()

    missing = []
    for module, install_name, purpose in PACKAGES:
        if module == "graphviz":
            works, why = graphviz_really_works()
            if works:
                print(f"  ok       {install_name:<16} {'':<10} {purpose}")
            else:
                missing.append((module, install_name))
                print(f"  MISSING  {install_name:<16} {'':<10} {why}")
            continue
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "")
            print(f"  ok       {install_name:<16} {version:<10} {purpose}")
        except Exception:
            missing.append((module, install_name))
            print(f"  MISSING  {install_name:<16} {'':<10} {purpose}")

    print()
    if not missing:
        print("All set. You are ready for class.")
        return 0

    hard = [name for module, name in missing if module not in OPTIONAL]
    soft = [(module, name) for module, name in missing if module in OPTIONAL]

    if soft:
        print("Missing, but you can still come to class:")
        for module, name in soft:
            print(f"  {name}: {SURVIVABLE[module]}")
        print()
    if hard:
        print("Missing, and needed. Install with the environment active:")
        print()
        print("    conda activate cml")
        print("    pip install " + " ".join(n for n in hard if n != "python-graphviz"))
        if "python-graphviz" in hard:
            print("    conda install -c conda-forge python-graphviz -y")
        print()
        print("Then run this check again.")
        print()
        print("Still stuck? Email me this entire output. It is usually enough to")
        print("diagnose the problem without any back and forth.")
        return 1

    print("Nothing required is missing. You are ready for class.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
