# <img src="assets/gavel-icon.png" alt="" width="42"> GAVEL Studio

## What GAVEL Studio does

GAVEL Studio helps you detect what a language model is doing in a conversation—for example, building romantic trust to push a purchase, drafting a phishing message, or reinforcing delusional thinking.

Unlike output filters that match text after it has been generated, GAVEL reads a model's internal activations. Lightweight probes score **Cognitive Elements (CEs)** such as `emotionally_engaging`, `payment_tools`, and `making_threat`; rules combine those signals into higher-level behavioral detections.

Studio brings the complete workflow into one interface:

* Browse reusable rules, CEs, and rule sets from the community-maintained [gavel-rules](https://github.com/Offensive-AI-Lab/gavel-rules) library ([browse it live](https://offensive-ai-lab.com/gavel/viewer/) without installing anything).
* Create a rule manually, assemble one from existing CEs, or generate a draft from a plain-English scenario.
* Train probes against the hidden states of a Hugging Face causal language model.
* Calibrate and evaluate detections on rule test sets.
* Monitor conversations in real time and inspect the CEs and rules that fire.
* Export a trained rule set or apply the same policy to another model.

<div align="center">
  <img src="assets/browse-rules.png" alt="Browsing community rules in GAVEL Studio" width="80%">
  <br><br>
  <img src="assets/ce-monitor.png" alt="Monitoring Cognitive Elements in GAVEL Studio" width="80%">
</div>

## Content advisory

Detecting harmful behavior requires examples of it. The rules and CE datasets available through Studio contain realistic **synthetic** conversations depicting scams and manipulation, threats, hate speech, and delusional or conspiratorial dialogue. This material exists solely to train and validate detectors—it reflects no one's views, and none of it describes real people or events.

## Quick start

The standard setup requires Docker with Compose; [Docker Desktop](https://www.docker.com/products/docker-desktop/) includes both.

```sh
git clone https://github.com/Offensive-AI-Lab/gavel-studio.git
cd gavel-studio
cp backend/.env.example backend/.env
```

These commands download GAVEL Studio, enter the project directory, and create its local configuration file.

Start GAVEL Studio:

```sh
docker compose --env-file backend/.env up --build
```

The first start downloads and builds the application dependencies, so it can take several minutes. Keep the command running, then open [http://localhost:5173](http://localhost:5173) in your browser. Later starts reuse the downloaded dependencies and are faster.

To stop Studio, press <kbd>Ctrl</kbd>+<kbd>C</kbd>, or run `docker compose down` if it is running in the background. Your projects remain in `backend/db/gavel.sqlite3`.

## Run without Docker

If you prefer running the servers directly, you need [Python 3.12+](https://www.python.org) and [Node.js 20+](https://nodejs.org). The backend's dependencies are listed in [`backend/requirements.txt`](backend/requirements.txt); the commands below install them with [uv](https://docs.astral.sh/uv/), but any Python package manager works the same way.

Start the backend:

```sh
cd backend
uv venv
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
uv run uvicorn main:app --port 8000
```

Installing `torch` separately from the CPU wheel index matters on Linux, where the default PyPI package is the multi-gigabyte CUDA build; this is the same CPU build the Docker image uses. On macOS and Windows the line is harmless. Skip it and install plain `torch` if you want local CUDA training.

Start the frontend in a second terminal:

```sh
cd frontend
npm install
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173). No configuration is required: the backend creates its SQLite database on first boot, and the two servers find each other on their default ports. To enable AI rule/CE generation, run `cp backend/.env.example backend/.env` and set `OPENAI_API_KEY` in it.

## Core concepts

* **Cognitive Element (CE):** one detectable concept, such as an emotional appeal, a request for payment, or a threat. A CE includes examples used to train its probe and data used to calibrate its detection threshold.
* **Rule:** one behavior to detect. A rule organizes CEs into named groups and applies a boolean condition to those groups.
* **Rule set:** a reusable collection of rules trained and evaluated against one language model.
* **Probe:** a lightweight classifier trained on a model's internal activations to score one or more CEs.

For a worked example of how CEs, groups, and rule conditions fit together, read the [GAVEL user guide](https://github.com/Offensive-AI-Lab/gavel-rules/blob/main/GUIDE.md).

## The ecosystem

| Component                                                               | Purpose                                                                                                          |
| ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **GAVEL Studio** (this repository)                                | Local application for authoring rule sets, training probes, evaluating detections, and monitoring conversations. |
| **[gavel-rules](https://github.com/Offensive-AI-Lab/gavel-rules)** | Source of truth for shared rules, CEs, rulesets, and their datasets ([live viewer](https://offensive-ai-lab.com/gavel/viewer/)). |
| **[GAVEL](https://github.com/Offensive-AI-Lab/gavel)**             | Activation-monitoring framework that underpins probe training and rule-based detection.                          |

Studio reads the public library into your local workspace. Contributions to shared rules and CEs are made by pull request to `gavel-rules`; Studio does not publish local drafts automatically.

## Configuration and compute

Studio works without API credentials for browsing, manual authoring, training, evaluation, and monitoring. Add an `OPENAI_API_KEY` to `backend/.env` to enable AI-assisted rule and dataset generation.

Training runs on the local CPU by default. Because Studio loads the target language model to collect activations, practical memory and compute requirements depend on that model. You can also use a local NVIDIA GPU with [`docker-compose.gpu.yml`](docker-compose.gpu.yml) or a remote worker described in [`gavel-gpu-worker/README.md`](gavel-gpu-worker/README.md).

Studio is designed as a single-user application bound to localhost. It has no account or access-control layer, so do not expose its backend directly to a shared or public network. Application data is stored in the local SQLite database at `backend/db/gavel.sqlite3`.

## Contributing

Issues and pull requests are welcome. Code changes should include the relevant backend or frontend tests; these tests run automatically on every pull request. To contribute a new public rule or CE, follow the contribution instructions in [gavel-rules](https://github.com/Offensive-AI-Lab/gavel-rules#contributing).

## Citation

If you use GAVEL in your research, please cite:

```bibtex
@inproceedings{rozenfeld2026gavel,
  title={GAVEL: Towards Rule-Based Safety Through Activation Monitoring},
  author={Rozenfeld, Shir and Pankajakshan, Rahul and Zloczower, Itay and Lenga, Eyal and Gressel, Gilad and Mirsky, Yisroel},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## License

GAVEL Studio is released under the license in [LICENSE](LICENSE).

## Contributors

GAVEL Studio is developed and maintained by the [Offensive AI Lab](https://offensive-ai-lab.com/).

### Principal investigator

* **Yisroel Mirsky** ([ymirsky](https://github.com/ymirsky))

### Research team

* **Shir Rozenfeld** ([shirozenfeld](https://github.com/shirozenfeld))
* **Rahul Pankajakshan** ([rahulgitsit](https://github.com/rahulgitsit))
* **Gilad Gressel** ([giladgressel](https://github.com/giladgressel))

### Software design and development

* **Ofek Avigezer** ([OfekAvi](https://github.com/OfekAvi))
* **Shahaf Har-Tsvi** ([hartsvis](https://github.com/hartsvis))
* **Shahar Navian** ([ShaharNavian](https://github.com/ShaharNavian))
* **Sean Shuhman** ([SeanSh1](https://github.com/SeanSh1))
* **Fadi Amon** ([FadiAmon](https://github.com/FadiAmon))

We also thank everyone who has contributed code, bug reports, testing, documentation, and feature suggestions.

## Acknowledgments

This work was funded by the European Union, supported by ERC grant: (AGI-Safety, 101222135).
Views and opinions expressed are however those of the author(s) only and do not necessarily reflect
those of the European Union or the European Research Council Executive Agency. Neither the
European Union nor the granting authority can be held responsible for them.
This work was also supported by the Israeli Ministry of Innovation Science and Technology (grant
number 1001948211).
