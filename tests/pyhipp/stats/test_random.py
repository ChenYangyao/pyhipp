from pyhipp.stats import random

def test_seed_sequence():
    for i in range(10):
        ss = random.SeedSequence(i)
        seeds = [ss.get_seed() for _ in range(5)]
        print(seeds)