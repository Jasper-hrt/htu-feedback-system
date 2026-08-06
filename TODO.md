# Deployment Readiness Fixes — HTU SRC System

- [x] 1. render.yaml: change startCommand to `gunicorn --worker-class eventlet -w 1 app:app`
- [x] 2. download_nltk_data.py: add `sentiwordnet` to NLTK_DATA list
- [x] 3. app.py `_ensure_nltk_data()`: add `sentiwordnet` + fix resource paths (`tokenizers/punkt`, `taggers/averaged_perceptron_tagger`)
- [x] 4. Verify app imports cleanly and sentiment pipeline works end-to-end
