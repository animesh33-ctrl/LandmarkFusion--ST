

from typing import List
import torch


def greedy_ctc_decode(log_probs: torch.Tensor,
                      blank_id: int = 0) -> List[List[int]]:
    
    if log_probs.dim() == 2:
        log_probs = log_probs.unsqueeze(0)

    B, T, C = log_probs.shape
    preds = log_probs.argmax(dim=-1)   # (B, T)
    results = []

    for b in range(B):
        seq = []
        prev = blank_id
        for t in range(T):
            tok = preds[b, t].item()
            if tok != blank_id and tok != prev:
                seq.append(tok)
            prev = tok
        results.append(seq)

    return results


def compute_wer(hypotheses: List[List[str]],
                references: List[List[str]]) -> float:
   
    total_dist = 0
    total_ref  = 0

    for hyp, ref in zip(hypotheses, references):
        n, m = len(ref), len(hyp)
        # DP table
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1): dp[i][0] = i
        for j in range(m + 1): dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref[i-1] == hyp[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j],   # deletion
                                        dp[i][j-1],    # insertion
                                        dp[i-1][j-1])  # substitution
        total_dist += dp[n][m]
        total_ref  += max(n, 1)

    return total_dist / total_ref
