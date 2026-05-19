"""
CS171 RSA Dataset Generator
32-bit and 64-bit only (CUDA modular exponentiation benchmark)
"""

import random
import time
import os
from math import gcd


# ============================================================
# SEED
# ============================================================

USE_RANDOM_SEED = False
FIXED_SEED = 67

SEED = int(time.time()) if USE_RANDOM_SEED else FIXED_SEED
random.seed(SEED)


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "data"

KEY_SIZES = [32, 64, 128, 512]

CASES_PER_SIZE = {
    32: 10000000,
    64: 1000000,
    128: 1000000,
    512: 2
}


# ============================================================
# MILLER–RABIN PRIMALITY TEST
# ============================================================

def is_probable_prime(n, rounds=20):
    if n < 2:
        return False

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    for p in small_primes:
        if n % p == 0:
            return n == p

    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    for _ in range(rounds):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False

    return True


# ============================================================
# PRIME GENERATION
# ============================================================

def generate_prime(bits):
    while True:
        p = random.getrandbits(bits)

        # ensure correct bit-length
        p |= (1 << (bits - 1))

        # ensure odd
        p |= 1

        if is_probable_prime(p):
            return p


# ============================================================
# RSA KEYPAIR
# ============================================================

def generate_rsa_keypair(bits):
    half = bits // 2

    while True:
        p = generate_prime(half)
        q = generate_prime(bits - half)

        if p == q:
            continue

        n = p * q
        phi = (p - 1) * (q - 1)

        if n.bit_length() < bits:
            continue

        return n, phi


# ============================================================
# EXPONENT
# ============================================================

def generate_exponent(bits, phi):
    while True:
        e = random.getrandbits(bits - 1)

        e |= (1 << (bits - 2))
        e |= 1

        if gcd(e, phi) == 1:
            return e


# ============================================================
# PLAINTEXT
# ============================================================

def generate_plaintext(n):
    for _ in range(10000):
        m = random.randrange(2, n - 1)
        if gcd(m, n) == 1:
            return m
    raise RuntimeError("Failed to generate plaintext")


# ============================================================
# SINGLE CASE
# ============================================================

def generate_case(bits):
    n, phi = generate_rsa_keypair(bits)

    e = generate_exponent(bits, phi)
    d = pow(e, -1, phi)

    m = generate_plaintext(n)

    c = pow(m, e, n)

    # verification
    assert pow(c, d, n) == m

    return m, e, n


# ============================================================
# PROGRESS
# ============================================================

def progress(i, total, elapsed):
    pct = 100 * i / total
    rate = i / elapsed if elapsed else 0
    eta = (total - i) / rate if rate else 0

    return f"{i}/{total} {pct:5.1f}% elapsed {elapsed:6.1f}s ETA {eta:6.1f}s"


# ============================================================
# MAIN GENERATION
# ============================================================

def generate_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    meta = []
    meta.append("# CS171 RSA Dataset")
    meta.append(f"# Seed: {SEED}")
    meta.append(f"# Key sizes: {KEY_SIZES}")

    total_cases = 0
    start_all = time.time()

    for bits in KEY_SIZES:
        count = CASES_PER_SIZE[bits]
        path = os.path.join(OUTPUT_DIR, f"dataset_{bits}bit_{count}cases.txt")

        print(f"\nGenerating {bits}-bit dataset ({count} cases)")

        start = time.time()

        with open(path, "w") as f:
            for i in range(1, count + 1):
                m, e, n = generate_case(bits)
                f.write(f"{m} {e} {n}\n")

                if i % 5000 == 0:
                    print(progress(i, count, time.time() - start), end="\r")

        elapsed = time.time() - start

        print(f"\nDone {bits}-bit in {elapsed:.1f}s")

        meta.append(f"{bits}-bit: {count} cases in {elapsed:.1f}s")

        total_cases += count

    total_time = time.time() - start_all

    meta.append(f"# Total cases: {total_cases}")
    meta.append(f"# Total time: {total_time:.1f}s")

    with open(os.path.join(OUTPUT_DIR, "metadata.txt"), "w") as f:
        f.write("\n".join(meta))


# ============================================================

if __name__ == "__main__":
    generate_dataset()