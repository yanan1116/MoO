

## Overview
This repo is a reproduction work based on `Mixture-of-Agents Enhances Large Language Model Capabilities, arXiv preprint arXiv:2406.04692` (original repo : https://github.com/togethercomputer/MoA)

Instead of testing MoA on benchmarks of MT-bench and Alpaca, which requires GPT-4 for evaluation. 
Here, this repo test MoA on math reasoning and QA tasks.

## Basec setting
Only one intermidiate layer is set for better ablation study.

Follow the executions to run it.
```
python -u moagen.py --rounds 1 --bench gsm # run on GSM8K
python -u moagen.py --rounds 1 --bench qa  # run on hotpotQA benchmark
```


## Enhanced with feedback
This repo also proposes an enhanced version of Mixture of Agents, where critics committee is used within each layer.
The feedback and critics of each reponse from each agent, is appended as additonal insights from the committee, which is also 
consist of a plenty of LLMs-driven agents.



## Empirical findings
MoA does not improve in math reasoning and hotpotQA benchmarks, regardless of the size of the LLMs.
Feel free to leave your discussions in this repo.



