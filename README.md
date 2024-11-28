# Lightweight MNIST Classifier

A lightweight CNN implementation for MNIST classification using PyTorch.

## Installation

This project uses `uv` as the package manager. First, install `uv`:

```
bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up the project:

```
bash
Clone the repository
git clone https://github.com/RupeshSatija/lightweight-mnist.git
cd lightweight-mnist
```

Create virtual environment and install dependencies
```
uv venv
source .venv/bin/activate 
```

Install dependencies
```
uv pip install -r requirements/requirements.txt
```
For development
```
uv pip install -r requirements/requirements-dev.txt
```
For testing
```
uv pip install -r requirements/requirements-test.txt
```

## Development

### Code Quality

This project uses several tools to maintain code quality:
- `black` for code formatting
- `isort` for import sorting
- `ruff` for linting
- `mypy` for type checking

Run the following commands before committing:

```
bash
black .
isort .
ruff check .
mypy src tests
```

### Testing

Run tests with:
bash
```
pytest tests/
```

## Usage
Train the model
```
python -m src.scripts.train
```
Evaluate the model
```
python -m src.scripts.evaluate
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.