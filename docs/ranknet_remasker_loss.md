# RankNet Remasker Loss

The remasker `g_phi` is used at sampling time to rank token positions by how likely they are to be wrong. The default BCE objective trains each position independently as a binary classifier, but the sampler ultimately uses the relative order of logits when it chooses the top positions to remask.

The RankNet-style loss trains that ranking behavior directly. For each sequence, positions labeled as errors are positives and positions labeled as correct are negatives. The loss compares every positive-negative pair inside the same sequence and encourages the error logit to be larger:

```text
L_b = mean_{i in P_b, j in N_b} softplus(score_j - score_i)
```

Here `P_b` is the set of error positions in sequence `b`, `N_b` is the set of correct positions, and `score_i` is the remasker logit for position `i`. If a sequence has no positive or no negative positions, it is skipped for the pairwise loss.

This matches the sampler's decision rule more closely than independent BCE: during guided star-shaped sampling, the remasker does not need calibrated probabilities as much as it needs a useful ordering of positions. High-scoring tokens are treated as more suspicious and are more likely to be selected by the Gumbel top-k remasking step.

In code, the loss is implemented by `compute_ranknet_pairwise_loss` in `remasker_train.py`. It uses the same labels as the selected remasker training strategy and respects `training.remasker_loss_only_new_tokens` through the loss mask.

To enable it, set:

```bash
training.remasker_use_ranknet_pairwise_loss=true
```

or use the provided script:

```bash
bash scripts/train/train_remasker_ranknet.sh
```
