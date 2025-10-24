ANONYMOUS SUBMISSION 


# Latent Representations and Data Augmentation for Citation Classification in Scientific Papers


Despite their remarkable success in many if not most natural language processing tasks, Large Language Models
(LLM) performance is very low for Citation Intent Classification in the few-shot settings. This task aims at recognizing
the intent (leverage, critic, use as motivation...) of the authors when they cite previous works in a scientific paper.
 We hence release a corpus composed of 10,786 silver-annotated sentences that we leverage to train a simple and generic classification pipeline that gives
state-of-the-art results on the ACL-ARC benchmark in the few-shots settings. The main advantage of our LLM-based
approach is its simplicity and genericity, which may hopefully be transposed at minimum cost to other datasets and
similar tasks found to be challenging for LLMs.

# How we address this ?
We assume that this poor performance of LLMs is due to massive class imbalance and data scarcity.


## Better representations

We analyze where in the LLM residual stream do the most relevant concepts for this task form and explore synthetic data
generation from a small number of labeled samples to alleviate these issues.
1- We extract activations for each layers for each sample in ACL ARC's train
    We do that in Extract_activations.ipynb

2- We then evaluate those activations in Eval_layers.ipynb

## Generate synthetic data

    1- The best dataset generated is Dataset 4. The methodology of the paper is based on it.
        the details and the code used to generate is in openrouter.ipynb
    2- We encode that synthetic data in evaluation.ipynb
    3- then it is evaluated also in evaluation.ipynb


