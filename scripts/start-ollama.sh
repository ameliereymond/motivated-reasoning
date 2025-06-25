#!/bin/bash

export OLLAMA_MODELS=/gscratch/clmbr/amelie/.cache/ollama/models
nohup ollama serve > ollama.log 2>&1 &