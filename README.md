# Causal Machine Learning: Setup

ECON 594, University of South Carolina, Fall 2026.

Lectures are **live Jupyter notebooks**, not PDFs. You will open the same notebooks I present
from, run the code yourself, and take notes in them.

This sets up one environment, `cml`, with everything the course needs, and connects your machine
to the course repository. Do it once.

Budget about 30 minutes, most of it download time.

> The slides that go with this guide are `Lectures/00-Course-Setup.pdf`. Same material, in the
> order I present it. Read either one.

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
git config --global user.email "you@email.sc.edu"
```

Use your real name and your university email. Check it took:

```bash
git config --global --list
```

You should see your `user.name` and `user.email` echoed back.

## Step 4. Make a GitHub account

**<https://github.com/signup>**

Free. Use your university email if you can, and pick a username you would not mind an employer
seeing, since this account tends to outlive the course. While you are there, the GitHub Student
Developer Pack is free with a `.edu` address and worth having.

GitHub will ask you to set up two factor authentication. Do it now rather than at the moment you
are trying to submit a problem set.

## Step 5. Fork the course repository

In a browser, go to the course repository:

**<https://github.com/Applied-Econometric-Methods/ECON594-Causal-Machine-Learning>**

1. Click **Fork**, top right.
2. Leave the name as it is. Click **Create fork**.
3. You now have `github.com/YOURNAME/ECON594-Causal-Machine-Learning`.

The page should say "forked from Applied-Econometric-Methods/ECON594-Causal-Machine-Learning" under the title.
That line is the link that makes syncing work later.

A **fork** is a copy of the whole repository under your own account. It is where your work lives
and it is what you submit problem sets from. You have no permission to write to my repository,
and you should not want it: forty students pushing to one repository is chaos.

## Step 6. Clone your fork

Open a conda aware terminal:

- **Windows:** Start menu, then **Anaconda Prompt**. Not PowerShell, not Command Prompt.
- **macOS / Linux:** any terminal.

You should see `(base)` at the start of the prompt. Move to wherever you keep coursework, then
clone **your fork**, not mine:

```bash
cd path/to/your/coursework
git clone https://github.com/YOURNAME/ECON594-Causal-Machine-Learning.git
cd ECON594-Causal-Machine-Learning
```

Replace `YOURNAME` with your GitHub username. If you would rather not type the URL, copy it off
your own fork's page under the green **Code** button.

```bash
ls        # Windows Anaconda Prompt: dir
```

## Step 7. Add my repository as `upstream`

Your clone already knows about your fork, under the name `origin`. Now tell it about mine, under
the name `upstream`, so you can pull my updates all term:

```bash
git remote add upstream https://github.com/Applied-Econometric-Methods/ECON594-Causal-Machine-Learning.git
git remote -v
```

You should see four lines: two for `origin` pointing at your fork, and two for `upstream`
pointing at mine.

**This is the step people skip.** Without it, `git pull` has no idea where my new material lives,
and in week 4 you will be downloading notebooks by hand.

## Step 8. Create and activate the environment

```bash
conda create -n cml python=3.11 -y
conda activate cml
```

Your prompt should now read `(cml)`.

**Run `conda activate cml` every time you open a new terminal for this course.** Forgetting is
the most common source of "it worked yesterday."

An environment is a private box of packages. Yours can hold pandas 2.3 while another project
holds pandas 1.5, and neither disturbs the other.

## Step 9. Install the analysis packages

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

## Step 10. Install JupyterLab and the course toolkit

```bash
pip install jupyterlab ipykernel variable-explorer jupyterlab_cell_enhancements
```

`ipykernel` is what actually runs your Python code. JupyterLab does not install it for you, and
without it Lab opens but has no kernel to run anything.

- **variable-explorer**: a panel showing your variables and their values as you go, similar to
  Stata's or RStudio's.
- **jupyterlab_cell_enhancements**: quality of life improvements for working in cells.

Confirm both registered:

```bash
jupyter labextension list
```

You want `variable-explorer` and `jupyterlab-cell-enhancements` both listed as `enabled ok`.

Then `pywrangling`, my own toolkit. It is not on PyPI, so pip installs it straight from GitHub,
which is one reason you installed Git:

```bash
pip install git+https://github.com/robertpettis/pywrangling.git
```

To pick up later updates, run the same command with `--upgrade --force-reinstall`.

## Step 11. Check that it all worked

One command tells you the state of everything you just installed. Run it from the course folder
with the environment active:

```bash
conda activate cml
python check_setup.py
```

You want a run of `ok` lines and a final `All set.` Anything marked `MISSING` names the package
that did not install, so you can rerun just that piece.

If graphviz is the only thing it complains about, you are still fine to come to class. The
notebooks will draw their diagrams with matplotlib instead.

This is also the thing to run if something breaks later in the term. Paste its full output into
an email and it is usually enough to diagnose the problem without any back and forth.

## Step 12. Launch

From inside the course folder:

```bash
jupyter lab
```

Because you launched it from the activated environment, Lab uses that environment's Python. The
kernel shows as **Python 3 (ipykernel)** in the top right. That is the right one.

Start Lab from the course folder, not from your desktop. The notebooks read their data through
paths like `./data/wage.csv`, and those only resolve if Lab was started from the right place.

Open a lecture notebook, run the first cell, and you are ready for class.

---

# Staying in sync during the semester

I add notebooks as we go, and I revise the ones already there: fixing errors, adding a section,
rebuilding a figure. So there are two things you need to be able to do, over and over.

## Getting my updates

Run this before most classes, from inside the course folder:

```bash
git pull upstream main
```

That fetches whatever I have changed and merges it into your copy. New notebooks appear, revised
ones update. Then push the result up to your own fork so it stays current too:

```bash
git push origin main
```

(`git pull` is two operations in one: `git fetch upstream` downloads my snapshots, and
`git merge upstream/main` folds them into your files.)

If the command line is fighting you, your fork's page on GitHub has a **Sync fork** button that
does the same thing in the browser. Click it, then run `git pull` to bring the result down to
your laptop.

### Why this works even though I change things constantly

I revise decks all term while you are working in the same folder. These do not collide, because
git merges file by file. I edit files in `Lectures/`; you add files in your own folder. Two
people editing different files is not a conflict, no matter how often either of us changes ours.

So a pull that arrives in the middle of your work updates my notebooks, leaves your files exactly
as they were, and needs nothing from you. The one way to break it is to edit my notebooks in
place, which is what the next section is about.

## The habit that avoids all the pain

**Never edit my notebooks in place. Work in a copy.**

Before you touch `06-Cross-Validation.ipynb`, copy it to `06-Cross-Validation-MYNOTES.ipynb` and
work in that.

- My original stays untouched, so `git pull` updates it cleanly forever.
- Your notes are a separate file that I never touch, so nothing of yours is ever overwritten.
- In class, run the original. Afterwards, work in your copy.

Notebooks are stored as one long machine readable file, and git merges them line by line. Two
people editing the same notebook produces a mess that is genuinely unpleasant to untangle. One
copied file avoids all of it.

## Handing in your work

Your fork is what you submit. Three commands, every time:

```bash
git add .
git commit -m "Problem set 1"
git push origin main
```

- `add` chooses what goes in the snapshot. The `.` means everything that changed.
- `commit` takes the snapshot, with a message.
- `push` sends it to your fork on GitHub.

Refresh your fork's page in the browser and your files are there. If you can see them on GitHub,
I can see them. If you cannot, I cannot, and it is not submitted.

## The five commands you will actually use

| command | what it does |
|---|---|
| `git status` | what has changed, and where you stand |
| `git pull upstream main` | get my latest material |
| `git add .` | mark your changes for the next snapshot |
| `git commit -m "..."` | take the snapshot |
| `git push origin main` | send it to your fork |

When you are lost, `git status` is always safe to run and usually tells you what to do next.

---

# Troubleshooting

**`jupyter: command not found`**
The environment is not active. Check your prompt reads `(cml)`, then `conda activate cml`.

**`git: command not found`**
Git is not installed or not on your PATH. Reinstall from <https://git-scm.com/downloads> and
open a new terminal.

**`ModuleNotFoundError`**
Almost always the wrong environment rather than a missing package. Check the prompt reads
`(cml)`, then rerun `python check_setup.py`.

**`git pull` says my local changes would be overwritten**
You edited a file that I also changed. First run `git status` to see where you stand. If the
edits it lists are ones you do not care about, throw them away and pull again:

```bash
git checkout -- 06-Cross-Validation.ipynb
git pull upstream main
```

If you do care about them, copy the file somewhere outside the course folder first, then do the
same. Nothing here is an emergency, and no situation requires deleting everything and starting
over.

**Commands run but versions look wrong**
You are probably in `base`. Check what your prompt says, then `conda activate cml`.

**Diagrams look plain, or `check_setup.py` says graphviz is not usable**
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

Then start again from Step 8.

**Still stuck**
Bring your laptop to office hours, or email me the exact command you ran and the complete error
message. A pasted error message usually takes under a minute to diagnose. Do not spend a weekend
on an install.
