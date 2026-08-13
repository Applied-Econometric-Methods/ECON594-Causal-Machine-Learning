# Causal Machine Learning: Setup

Lectures are **live Jupyter notebooks**, not PDFs. You will open the same notebooks I present
from, run the code yourself, and take notes in them.

This sets up one environment, `cml`, with everything the course needs. Do it once.

Budget about 30 minutes, most of it download time.

---

## Step 1. Install Anaconda

Download the **Anaconda Distribution** installer:

**<https://www.anaconda.com/download>**

Run it and accept the defaults, with one exception:

- **Windows:** install for "Just Me," and do **not** check "Add Anaconda to my PATH." It is
  offered, it is marked not recommended, and it causes conflicts later. You will use the
  Anaconda Prompt instead.

If you already have Anaconda or Miniconda, skip this step.

## Step 2. Install Git

Download and install Git:

**<https://git-scm.com/downloads>**

Accept the defaults. On Windows this also installs "Git Bash," which you can ignore.

## Step 3. Tell Git who you are

Git stamps your name and email on anything you commit. Set them once, globally:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

Use your real name and your university email. Check it took:

```bash
git config --global --list
```

You should see your `user.name` and `user.email` echoed back.

## Step 4. Open a conda-aware terminal

- **Windows:** Start menu → **Anaconda Prompt**. Not PowerShell, not Command Prompt.
- **macOS / Linux:** any terminal.

You should see `(base)` at the start of the prompt.

## Step 5. Clone the course repository

Move to wherever you keep coursework, then clone the repo. This is where the lecture notebooks
and assignments live:

```bash
cd path/to/your/coursework
git clone https://github.com/robertpettis/ECON594-Causal-Machine-Learning.git
cd ECON594-Causal-Machine-Learning
```

You should now see `Lectures`, `Assignments`, and `Syllabus` folders:

```bash
ls        # Windows Anaconda Prompt: dir
```

I will add notebooks as we go, so run this from inside the folder before most classes to pull
down new material:

```bash
git pull
```

## Step 6. Create and activate the environment

```bash
conda create -n cml python=3.11 -y
conda activate cml
```

Your prompt should now read `(cml)`.

**Run `conda activate cml` every time you open a new terminal for this course.** Forgetting is
the most common source of "it worked yesterday."

## Step 7. Install the analysis packages

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn statsmodels linearmodels networkx imageio-ffmpeg fklearn
```

Every package on that line installs the same way on Windows, Mac, and Linux, with no extra
system software.

The neural network lectures use **PyTorch**. It is a much larger download, several hundred
megabytes, so give it its own line and a little patience:

```bash
pip install torch
```

That installs the CPU build, which is all this course needs. You do not need a graphics card.

(`imageio-ffmpeg` supplies the video encoder the lecture notebooks use to render animations
inline. It bundles its own binary, so there is nothing else to install.)

Now one more, for the causal diagrams. This one is `conda`, not `pip`, and that matters:

```bash
conda install -c conda-forge python-graphviz -y
```

Graphviz is a drawing program with a Python wrapper around it. Installing it with `pip` gets
you the wrapper only, and then diagrams fail with a confusing error about a missing `dot`
program. The conda-forge version above installs the drawing program **and** the wrapper
together, on every operating system, which is why we use it.

If this step gives you trouble, keep going anyway. The lecture notebooks detect a missing
graphviz and fall back to drawing the diagrams with matplotlib instead. They look plainer, but
nothing breaks and no cell fails.

## Step 8. Install JupyterLab and extensions

```bash
pip install jupyterlab ipykernel variable-explorer jupyterlab_cell_enhancements
```

`ipykernel` is what actually runs your Python code. JupyterLab does not install it for you, and
without it Lab opens but has no kernel to run anything.

- **variable-explorer**: a panel showing your variables and their values as you go, similar to
  Stata's or RStudio's.
- **jupyterlab_cell_enhancements**: quality-of-life improvements for working in cells.

Confirm both registered:

```bash
jupyter labextension list
```

You want `variable-explorer` and `jupyterlab-cell-enhancements` both listed as `enabled ok`.

## Step 9. Install `pywrangling`

`pywrangling` is my own toolkit. It is not on PyPI, so pip installs it straight from GitHub,
which is why you installed Git:

```bash
pip install git+https://github.com/robertpettis/pywrangling.git
```

To pick up later updates, run the same command with `--upgrade --force-reinstall`.

Verify:

```bash
python -c "import pywrangling; print('pywrangling ok')"
```

## Step 10. Check that it all worked

One command tells you the state of everything you just installed. Run it from the `Lectures`
folder of the course repository:

```bash
cd Lectures
python -c "import course_helpers; course_helpers.check_setup()"
```

You want a run of `ok` lines and a final `All set.` Anything marked `MISSING` names the package
that did not install, so you can rerun just that piece.

If graphviz is the only thing it complains about, you are still fine to come to class. The
notebooks will draw their diagrams with matplotlib instead.

This is also the thing to run if something breaks later in the term. Paste its full output into
an email and it is usually enough to diagnose the problem without any back and forth.

## Step 11. Launch

From inside the course folder:

```bash
jupyter lab
```

Because you launched it from the activated environment, Lab uses that environment's Python. The
kernel shows as **Python 3 (ipykernel)** in the top right. That is the right one.

Open a lecture notebook, run the first cell, and you are ready for class.

---

# Troubleshooting

**`jupyter: command not found`**
The environment is not active. Check your prompt reads `(cml)`, then `conda activate cml`.

**`git: command not found`**
Git is not installed or not on your PATH. Reinstall from <https://git-scm.com/downloads> and
open a new terminal.

**`git pull` says I have local changes**
You edited a lecture notebook in place. Copy your version somewhere safe, then discard the
change: `git checkout -- <file>`. Better habit: duplicate a notebook before working in it, and
leave the originals alone so `git pull` stays clean.

**Commands run but versions look wrong**
You are probably in `base`. Check what your prompt says, then `conda activate cml`.

**Diagrams look plain, or `check_setup` says graphviz is not usable**
Graphviz did not install. In the activated `cml` environment run:

```bash
conda install -c conda-forge python-graphviz -y
```

If you previously ran `pip install graphviz`, undo it first with `pip uninstall graphviz -y`,
since the pip package shadows the conda one and provides no drawing program. Nothing here is
urgent: the notebooks fall back to matplotlib diagrams and run fine either way.

**Everything is broken and I have been at this an hour**
Delete the environment and redo it. This takes five minutes:

```bash
conda deactivate
conda env remove -n cml
```

Then start again from Step 6.

**Still stuck**
Bring your laptop to office hours, or email me the exact command you ran and the complete error
message. A pasted traceback usually takes under a minute to diagnose.
