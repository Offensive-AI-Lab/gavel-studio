# Reference implementation — do not edit

This directory is a **verbatim copy of the published GAVEL paper code** from
[github.com/Offensive-AI-Lab/gavel](https://github.com/Offensive-AI-Lab/gavel).

**Do not change any file in it.** The copy is kept unmodified so that GAVEL
Studio produces exactly the numbers the paper describes: threshold sweeps,
Youden-J, use-case detection, AUC, and sliding-window inference. Editing a file
here breaks that guarantee silently.

## Where your change belongs instead

In `backend/evaluation/adapter.py`. Everything Studio adds on top of this code —
reading data from the database, connecting it to the API — lives there.

If something in this code is wrong, you have two options:

1. Fix it in the upstream project and refresh this whole copy, or
2. Override the behavior in the adapter.

## How it is used

Nothing here is started on its own. The backend imports it.
[`__init__.py`](__init__.py) lets the copy answer to its original `gavel.*`
import names, which is why these files work without being edited.

## How to check it still works

From `backend/`, with the development dependencies installed
(`pip install -r requirements.txt -r requirements-dev.txt`):

```sh
pytest tests/integration/test_reference_parity.py
```

That test sends fixed inputs through the reference functions and compares the
results with values worked out by hand. Run it after refreshing this copy.

## Refreshing this copy

Copy `gavel/{evaluation,training,preprocessing,utils,models,config.py}` from the
upstream project over this directory, then run the test above. No edits inside
this tree are needed.

## Learn more

* [Repository README](../../../README.md) — what GAVEL Studio is and how to run it.
* [GAVEL](https://github.com/Offensive-AI-Lab/gavel) — the paper code this copy comes from.
