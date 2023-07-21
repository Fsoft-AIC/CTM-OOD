# Class Typical Matching for OOD detection

This is a PyTorch implementation of the paper [A Cosine Similarity-based Method for Out-of-Distribution Detection](https://arxiv.org/abs/2306.14920). 
The poster is available [here](./Poster.pdf).

**How to run the code:**
1. Install requirements. Run `pip install -r requirements.txt` to install dependencies.
2. Setup environment variables (dataset and checkpoints paths) to `.env` file. See `.env.example` for refference.
3. Run scripts from `experiment/Benchmark` folder. We use hydra for configuration. See `experiment/Benchmark/configs` for `net` and and `detector` options. 

