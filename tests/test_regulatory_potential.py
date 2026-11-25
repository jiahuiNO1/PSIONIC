import numpy as np
import pytest

# tf_target_helpers imports the full single-cell stack (scanpy, pyranges, ...)
pytest.importorskip("pyranges")
from psionic.tf_target_helpers import RP_Simple  # noqa: E402

DECAY = 10_000
TSS = 1_000_000


def scores(peaks):
    """RP scores between one gene (chr1, TSS at 1 Mb) and a list of (chrom, center) peaks."""
    genes_info = [["1", TSS, 1, 0]]                          # [chrom, tss, 1, gene index]
    peaks_info = [[c, float(p), 0, i] for i, (c, p) in enumerate(peaks)]
    return RP_Simple(peaks_info, genes_info, DECAY).toarray()[0]


def test_scores_halve_every_decay_distance():
    s = scores([("1", TSS), ("1", TSS + DECAY), ("1", TSS - 2 * DECAY)])
    np.testing.assert_allclose(s, [1.0, 0.5, 0.25])


def test_distant_and_other_chromosome_peaks_score_zero():
    s = scores([("1", TSS + 16 * DECAY), ("2", TSS)])
    np.testing.assert_allclose(s, [0.0, 0.0])
