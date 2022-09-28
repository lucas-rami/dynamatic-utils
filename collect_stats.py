import os
from dataclasses import dataclass
from typing import Dict, Final

PATH: Final[str] = os.path.join("benchmarks")


@dataclass(frozen=True)
class IRStats:
    n_blocks: int
    n_instructions: int

    @staticmethod
    def from_file(fp: str) -> "IRStats":
        with open(fp, "r") as f:
            n_blocks = int(f.readline())
            n_instructions = int(f.readline())
            return IRStats(n_blocks, n_instructions)


@dataclass(frozen=True)
class BenchStats:
    llvm: IRStats
    mlir: IRStats

    @staticmethod
    def from_file(bench: str) -> "BenchStats":
        llvm = IRStats.from_file(path_llvm(bench))
        mlir = IRStats.from_file(path_mlir(bench))
        return BenchStats(llvm, mlir)


def path_llvm(name: str) -> str:
    return os.path.join(PATH, name, "llvm", "stats.txt")


def path_mlir(name: str) -> str:
    return os.path.join(PATH, name, "mlir", "stats.txt")


def run() -> None:
    benchmarks = os.listdir(PATH)
    print(f"Detected {len(benchmarks)} benchmarks")

    print(f"Parsing statistics")
    n_fail: int = 0
    stats: Dict[str, BenchStats] = {}
    for bench in benchmarks:
        try:
            stats[bench] = BenchStats.from_file(bench)
        except:
            n_fail += 1
    print(f"\t{n_fail}/{len(benchmarks)} benchmarks failed to parse")


if __name__ == "__main__":
    run()
