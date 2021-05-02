# Mappings between unicode (hex and int), and initial, medial, and final labels


def imf2int(initial, medial, final):
    "Initial, medial, and final index to unicode character."
    return 44032 + (21 * 28) * initial + 28 * medial + final


def imf2idx(initial, medial, final):
    "Initial, medial, and final index to dataset index."
    return (21 * 28) * initial + 28 * medial + final


def idx2imf(idx):
    "Initial, medial, and final index to dataset index."
    final = idx % 28
    idx = idx // 28
    medial = idx % 21
    initial = idx // 21
    return initial, medial, final


def imf2hex(initial, medial, final):
    return hex(imf2int(initial, medial, final))


def int2imf(unicode_int):
    decimal = int(unicode_int)
    decimal -= 44032
    final = decimal % 28
    medial = (decimal // 28) % 21
    initial = (decimal // (28 * 21)) % 19
    return initial, medial, final


def hex2imf(hex_string):
    decimal = int(hex_string, 16)
    return int2imf(decimal)
