# CI/CD Pipelines for Machine Learning Engineering

## 1. What CI/CD Means

CI/CD is the system that lets an engineering team ship code safely and repeatedly without manually running tests, building packages, deploying services, or checking production health every time.

For Machine Learning Engineering (MLE), CI/CD is especially important because you often ship both:

1. **Software code**
   - APIs
   - Batch jobs
   - Feature pipelines
   - Model serving code

2. **ML artifacts**
   - Models
   - Feature transformations
   - Configs
   - Data validation rules
   - Evaluation reports

---

## 2. CI = Continuous Integration

Continuous Integration means every time you push code or open a pull request, the system automatically checks whether the code is safe to merge.

Typical CI checks include:

- Unit tests
- Integration tests
- Lint checks
- Type checks
- Docker build
- Small model training smoke test
- Data validation checks

The goal is to catch problems before merging code.

Example commands:

```bash
pytest tests/
ruff check .
mypy src/
```

---

## 3. CD = Continuous Delivery / Continuous Deployment

After code passes CI, CD handles releasing it.

### Continuous Delivery

The code is ready to deploy, but a human approves the final release.

### Continuous Deployment

The code is automatically deployed to production after passing checks.

In ML systems, teams often prefer **continuous delivery** rather than fully automatic deployment, because model changes can affect business metrics and often need extra validation.

---

## 4. Simple CI/CD Pipeline Example

Imagine you build a retention prediction model service.

A simplified pipeline could look like this:

```text
Developer pushes code
        ↓
Run CI checks
        ↓
Run unit tests
        ↓
Run integration tests
        ↓
Build Docker image
        ↓
Push image to registry
        ↓
Deploy to staging
        ↓
Run smoke tests
        ↓
Approve production deployment
        ↓
Deploy to production
        ↓
Monitor latency, errors, prediction distribution
```

---

## 5. Why CI/CD Matters for MLE

As a Data Scientist, you may be used to working in notebooks:

```text
Load data → train model → evaluate → share result
```

As an MLE, you need to think:

```text
Can this run every day?
Can this fail safely?
Can someone else reproduce it?
Can we deploy it?
Can we roll back if it breaks?
Can we monitor it after launch?
```

CI/CD is the system that makes this possible.

---

# MLE-Specific CI/CD Pipeline

## Stage 1: Code Checks

These ensure the codebase is healthy.

Common checks:

- Formatting: `black`, `isort`
- Linting: `ruff`, `flake8`
- Type checking: `mypy`, `pyright`
- Unit tests: `pytest`

Example:

```bash
pytest tests/
ruff check .
mypy src/
```

---

## Stage 2: Data Validation

ML pipelines often break because data changes.

Important checks:

- Are required columns present?
- Are data types correct?
- Are null rates acceptable?
- Are feature distributions reasonable?
- Did the label distribution shift?
- Are values within expected ranges?

Example checks:

```python
assert "user_id" in df.columns
assert df["label"].isin([0, 1]).all()
assert df["age"].isna().mean() < 0.05
assert df["prediction_feature"].between(0, 1).all()
```

Popular tools:

- Great Expectations
- TensorFlow Data Validation
- Pandera
- Custom validation scripts

---

## Stage 3: Training Pipeline Smoke Test

You usually do not train the full model inside CI because it may be expensive.

Instead, run a smoke test:

```text
Can the training script run on a tiny sample?
Does it produce a model artifact?
Does evaluation finish?
Does the model output valid predictions?
```

Example:

```bash
python train.py --sample_size 1000 --max_epochs 1
python evaluate.py --model_path artifacts/model.pkl
```

The goal is not to train a good model. The goal is to make sure the pipeline does not break.

---

## Stage 4: Model Evaluation Gate

Before deploying a model, compare it against the current production model.

Example checks:

- New model AUC >= production AUC - tolerance
- New model log loss <= production log loss + tolerance
- Prediction distribution is not too different from production
- No major segment regression
- Latency is acceptable

Example:

```python
if new_auc < prod_auc - 0.005:
    raise ValueError("Model quality regressed too much")
```

This is highly relevant to MLE work because a model can pass software tests but still degrade business performance.

---

## Stage 5: Build Artifact

For ML, the deployable artifact could be:

- Docker image
- Model file
- Feature transformation pipeline
- Model config
- Inference service
- Batch scoring job

Example Docker build:

```bash
docker build -t retention-model-service:latest .
```

---

## Stage 6: Deploy to Staging

Staging is a production-like environment where you validate the service before production.

You should test:

- Can the service start?
- Can it load the model?
- Can it return predictions?
- Does it meet latency requirements?
- Does it handle bad input?
- Are prediction values valid?

Example smoke test:

```python
import requests

payload = {
    "onboarding_completed": 1,
    "messages_to_team_first_14d": 3,
    "roles_joined_first_30d": 2
}

r = requests.post("https://staging-model-service/predict", json=payload)

assert r.status_code == 200
assert 0 <= r.json()["score"] <= 1
```

---

## Stage 7: Production Deployment

Common deployment patterns include:

### Blue-Green Deployment

You keep two environments:

```text
blue = current production
green = new version
```

When the green environment is healthy, you switch traffic from blue to green.

This is good for quick rollback.

---

### Canary Deployment

Send a small percentage of traffic to the new model first:

```text
1% traffic → 5% → 25% → 50% → 100%
```

Monitor metrics at each step.

This is common for model serving.

---

### Shadow Deployment

The new model receives production traffic but does not affect users.

```text
Production model makes the real decision
Shadow model only logs predictions
```

This is useful when you want to compare predictions safely before launch.

---

# Example GitHub Actions CI Pipeline

```yaml
name: CI

on:
  pull_request:
  push:
    branches:
      - main

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Check out repo
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run lint
        run: |
          ruff check .

      - name: Run tests
        run: |
          pytest tests/

      - name: Run training smoke test
        run: |
          python train.py --sample_size 1000 --max_epochs 1

      - name: Run evaluation smoke test
        run: |
          python evaluate.py --model_path artifacts/model.pkl
```

This means every pull request automatically checks whether your ML code still works.

---

# Example CD Pipeline

```yaml
name: Deploy

on:
  push:
    branches:
      - main

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Check out repo
        uses: actions/checkout@v4

      - name: Build Docker image
        run: |
          docker build -t retention-model-service:${{ github.sha }} .

      - name: Push Docker image
        run: |
          docker push retention-model-service:${{ github.sha }}

      - name: Deploy to staging
        run: |
          ./scripts/deploy_staging.sh retention-model-service:${{ github.sha }}

      - name: Run staging smoke test
        run: |
          python tests/smoke_test_staging.py

      - name: Deploy to production
        run: |
          ./scripts/deploy_prod.sh retention-model-service:${{ github.sha }}
```

In real companies, this may use:

- GitHub Actions
- GitLab CI
- CircleCI
- Jenkins
- Buildkite
- Argo CD
- Spinnaker
- Kubernetes
- Terraform
- Docker
- AWS / GCP / Azure

---

# Interview Explanation

A strong MLE interview answer:

> For CI/CD in ML systems, I think about it as a safety and automation layer around the full model lifecycle. CI ensures that every code change passes unit tests, integration tests, data validation checks, and training or inference smoke tests. CD then packages the model or service, deploys it to staging, runs smoke tests, and gradually releases it to production through canary, shadow, or blue-green deployment. For ML specifically, I would also add model-quality gates, data drift checks, prediction-distribution monitoring, and rollback mechanisms because a model can pass software tests but still degrade business performance.

---

# Key Difference Between Software CI/CD and ML CI/CD

For normal software:

```text
Same code + same input = same output
```

For ML:

```text
Same code + different data = different model
Same model + shifted data = different behavior
```

Therefore, ML CI/CD needs extra checks:

- Data quality
- Feature consistency
- Model quality
- Prediction distribution
- Segment performance
- Drift
- Business metric impact

This is the MLE mindset.

---

# What to Learn Next for an MLE Pivot

Recommended stack:

- Git + GitHub Actions
- Docker
- pytest
- FastAPI model serving
- Kubernetes basics
- MLflow or Weights & Biases for model tracking
- Airflow / Dagster for pipelines
- Great Expectations or Pandera for data validation
- Canary / shadow deployment concepts
- Monitoring: logs, metrics, alerts

---

# Suggested Hands-On Project

Build a small retention prediction model and productionize it:

1. Train a simple model.
2. Serve it with FastAPI.
3. Containerize it with Docker.
4. Add unit tests and integration tests with pytest.
5. Add data validation checks.
6. Set up GitHub Actions CI.
7. Add a training smoke test.
8. Add a model evaluation gate.
9. Deploy to a cloud service or local Kubernetes.
10. Add monitoring for latency, error rate, and prediction distribution.

This kind of project is a strong portfolio piece for an MLE transition.
