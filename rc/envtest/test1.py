import argparse

p = argparse.ArgumentParser()
p.add_argument("--test_argument", default='hello world')

preargs = p.parse_args()

