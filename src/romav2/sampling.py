"""Tensor-only correspondence sampling, independent of the neural architecture."""

import torch

from .geometry import bhwc_grid_sample, get_normalized_grid


def sample_predictions(
    preds: dict[str, torch.Tensor], num_corresp: int, *, bidirectional: bool
):
    if num_corresp < 0:
        raise ValueError("num_corresp must be non-negative")
    warp = preds["warp_AB"]
    confidence_AB = preds["overlap_AB"]
    precision_AB = preds["precision_AB"] if "precision_AB" in preds else None

    warp = warp[0]
    confidence_AB = confidence_AB[0].reshape(-1)
    if precision_AB is not None:
        precision_AB = precision_AB[0]

    H_A, W_A, two = warp.shape
    grid = get_normalized_grid(1, H_A, W_A, overload_device=warp.device)[0]
    matches_AB = torch.cat((grid, warp), dim=-1).reshape(-1, 4)
    if bidirectional:
        confidence_BA = preds["overlap_BA"]
        warp_BA = preds["warp_BA"]

        precision_BA = preds["precision_BA"] if "precision_BA" in preds else None
        warp_BA = warp_BA[0]
        confidence_BA = confidence_BA[0]
        if precision_BA is not None:
            precision_BA = precision_BA[0]

        if precision_BA is not None and precision_AB is not None:
            precision_A = bhwc_grid_sample(
                precision_BA[None].reshape(1, H_A, W_A, -1),
                warp[None],
                mode="bilinear",
                align_corners=False,
            ).reshape(H_A, W_A, 2, 2)
            precision_B = bhwc_grid_sample(
                precision_AB[None].reshape(1, H_A, W_A, -1),
                warp_BA[None],
                mode="bilinear",
                align_corners=False,
            ).reshape(H_A, W_A, 2, 2)
            precision_fwd = torch.stack((precision_A, precision_AB), dim=-3).reshape(
                -1, 2, 2, 2
            )
            precision_bwd = torch.stack((precision_BA, precision_B), dim=-3).reshape(
                -1, 2, 2, 2
            )
            precision = torch.cat((precision_fwd, precision_bwd), dim=0)
        else:
            precision = None
        grid = get_normalized_grid(1, H_A, W_A, overload_device=warp.device)[0]
        matches_BA = torch.cat((warp_BA, grid), dim=-1).reshape(-1, 4)
        confidence = torch.cat(
            (confidence_AB.reshape(-1), confidence_BA.reshape(-1)), dim=0
        )
        matches = torch.cat((matches_AB, matches_BA), dim=0)
    else:
        matches = matches_AB
        confidence = confidence_AB.reshape(-1)
        precision = precision_AB.reshape(-1, 2, 2) if precision_AB is not None else None

    expansion_factor = 4
    valid = (
        torch.isfinite(matches).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence > 0)
        & matches.abs().amax(dim=-1).le(1 - 1 / H_A)
    )
    confidence = torch.where(valid, confidence, torch.zeros_like(confidence))
    count = min(expansion_factor * num_corresp, int(valid.sum().item()))
    if count == 0:
        return matches[:0], confidence[:0], None, None
    corresp_inds = torch.multinomial(confidence, count, replacement=False)
    sampled_matches = matches[corresp_inds]
    sampled_confidence = confidence[corresp_inds]
    if precision is not None:
        sampled_precision = precision[corresp_inds]
    else:
        sampled_precision = None
    density = kde(sampled_matches)

    p = 1 / (density.float() + 1)
    p[density < 10] = 1e-7
    balanced_samples = torch.multinomial(
        p, num_samples=min(num_corresp, len(sampled_confidence)), replacement=False
    )
    return (
        sampled_matches[balanced_samples],
        sampled_confidence[balanced_samples],
        sampled_precision[balanced_samples][:, 0]
        if sampled_precision is not None
        else None,
        sampled_precision[balanced_samples][:, 1]
        if sampled_precision is not None
        else None,
    )


def kde(x: torch.Tensor, std: float = 0.1, half: bool = True) -> torch.Tensor:
    if half and x.device.type == "cuda":
        x = x.half()
    else:
        x = x.float()
    # A full NxN distance matrix can exhaust GPU/CPU memory for 5000 matches.
    return torch.cat(
        [
            (-(torch.cdist(chunk, x) ** 2) / (2 * std**2))
            .exp()
            .sum(dim=-1, dtype=torch.float32)
            for chunk in x.split(512)
        ]
    )
