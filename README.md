# inkgraph

Inkgraph is a small, browser-first studio for turning graphs, diagrams, and text into expressive ink illustrations. The prototype lives in `index.html`, `styles.css`, and `app.js` with no build step.

## Run

Open `index.html` in a browser. The studio supports source tabs, recent examples, ink character selection, density and wobble controls, zoom, and SVG export.

## Test

```sh
node app.test.js
```

The tests use BDD-style scenarios and Node's built-in assertions, keeping the initial test surface dependency-free. The UI state is explicit and the export boundary is isolated in `app.js`, leaving room to add a parser and renderer adapter without coupling the controls to the drawing engine.

## Login and languages

The UI supports English, Simplified Chinese, and Japanese. The language choice is persisted in browser storage and covers the main studio controls. The Google Translate widget is also included, which exposes Google's current full supported-language catalog rather than a fixed local list.

Google login uses Google's browser Identity Services. To enable it, create an OAuth web client in Google Cloud, add the authorized origin for the host serving `index.html`, then place its public client ID in the empty `google-client-id` meta tag. The client ID is public configuration; never place a client secret in this repository. Without a client ID, the button reports that setup is required instead of simulating authentication.

The local login dialog calls `/register` and `/login` on the Python service. Passwords are salted and hashed with PBKDF2-HMAC-SHA256 (210,000 rounds), sessions are signed with an expiring HttpOnly cookie, malformed input is rejected, login errors are intentionally generic, and requests are rate limited per client IP. The prototype user store is in memory and is cleared when the service restarts; use a managed database, HTTPS, secret injection, CSRF protection, audit logging, and account recovery before production.

Serve the app and API from the same origin for local login, for example by placing the static files behind your normal web server and proxying `/login`, `/register`, and `/transform` to `ink_service.py`. Opening the HTML directly with `file://` will still show the UI, but browser fetches to local auth will not work.

## ML dry run

The repository now includes a runnable, dependency-free reference pipeline in `ink_ml.py`:

- **RNN** encodes the source sequence and predicts ink density, wobble, and texture.
- **GAN** proposes style controls, scores real/fake style vectors, and trains both discriminator and generator signals.
- **DQN** selects a transformation action such as `add-wash` or `add-labels` using replay memory, discounting, and a slowly blended target value.
- **Dataset** is a reproducible synthetic starter set generated from graph prompts. Replace `make_dataset()` with licensed graph/diagram pairs before production training.
- **Evaluation** creates a deterministic train/validation split and reports holdout mean-squared error on every epoch.
- **Ingestion** accepts licensed JSONL records with `source`, `family`, and optional `target` fields through `load_jsonl_dataset()`.

Run the training simulation:

```sh
python3 train_dry_run.py --epochs 3 --dataset-size 24
python3 -m unittest -v test_ink_ml.py test_auth.py
```

Run the local inference service and call it with `curl`:

```sh
python3 ink_service.py
curl -X POST http://127.0.0.1:8765/transform \
	-H 'Content-Type: application/json' \
	-d '{"source":"research -> sketch -> prototype -> share"}'
```

This is an honest dry-run baseline, not a production-quality neural renderer: it validates contracts, training orchestration, and deployment shape without hiding missing GPU/model dependencies. A production upgrade should use PyTorch, a licensed dataset of source SVGs plus style targets, held-out evaluation, and a real raster/vector renderer.
