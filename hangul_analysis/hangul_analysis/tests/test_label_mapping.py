import numpy as np

from hangul import label_mapping


def test_invert_imf():
    """Check that imf indices are mapped back to themselves."""
    rng = np.random
    for ii in range(100):
        imf = (rng.randint(19), rng.randint(21), rng.randint(28))
        assert imf == label_mapping.int2imf(label_mapping.imf2int(*imf))
        assert imf == label_mapping.hex2imf(label_mapping.imf2hex(*imf))
        assert imf == label_mapping.idx2imf(label_mapping.imf2idx(*imf))


def test_invert_int():
    """Check that int indices are mapped back to themselves."""
    rng = np.random
    for ii in range(100):
        n = rng.randint(44032, 55204)
        assert n == label_mapping.imf2int(*label_mapping.int2imf(n))
