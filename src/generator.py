import random
from pathlib import Path


def main() -> None:
    """
    Generate synthetic DNA reads and write them to both FASTA and plain-text files.

    - Length of each read: 200 bp (based on a seeded template).
    - Number of reads: 10,001.
    - Output:
        * sequences.txt (FASTA-like, used by C++ benchmark)
        * sequences.fa  (FASTA, used by SEQ benchmark)
    """

    random.seed(10)
    bases = ["A", "C", "T", "G"]

    # Build a deterministic 200 bp seed sequence.
    seed = "".join(bases[random.randint(0, 3)] for _ in range(200))

    out_txt = Path("sequences.txt")
    out_fa = Path("sequences.fa")

    with out_txt.open("w") as f_txt, out_fa.open("w") as f_fa:
        for i in range(10001):
            header = f">read{i}\n"
            f_txt.write(header)
            f_fa.write(header)

            seq = list(seed)
            # Introduce a random number of mutations per read.
            freq = random.randint(160, 180)
            for _ in range(freq):
                seq[random.randint(0, 199)] = bases[random.randint(0, 3)]

            s = "".join(seq)
            f_txt.write(s + "\n")
            f_fa.write(s + "\n")


if __name__ == "__main__":
    main()
