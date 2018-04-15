"""..."""

import os
from argparse import ArgumentParser

ROOT = os.path.dirname(os.path.abspath(__name__))


def collect():
    with open(os.path.join(ROOT, "data", "ruwikiruscorpora_0_300_20.bin"), "w") as f1:
        for i in range(10):
            with open(f1.name + ".part{}".format(i)) as f2:
                f1.write(f2.read())

            os.remove(f2.name)


def split():
    with open(os.path.join(ROOT, "data", "ruwikiruscorpora_0_300_20.bin")) as f1:
        f1_size = os.path.getsize(f1.name)
        for i, _ in enumerate(range(0, f1_size, 50*2**20)):
            with open(f1.name + ".part{}".format(i), "w") as f2:
                f2.write(f1.read(50*2**20))

    os.remove(f1.name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("command", choices=["collect", "split"])
    args = parser.parse_args()

    if args.command == "collect":
        collect()
    elif args.command == "split":
        split()
