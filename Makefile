install:
	pip install -r requirements.txt

run:
	uvicorn src.api:app

eval:
	python tests/test_eval.py

adversarial:
	python tests/test_adversarial.py

test:
	python tests/test_eval.py
	python tests/test_adversarial.py