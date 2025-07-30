#!/usr/bin/env bash
mkdir -p data/raw
kaggle datasets download -d fedesoriano/stroke-prediction-dataset \
  --path data/raw --unzip