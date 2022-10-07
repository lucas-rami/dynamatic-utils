import os
import json
from dataclasses import dataclass

from typing import ClassVar, Set, Mapping, Final, Dict

Stats = Mapping[str, "BenchStats"]


@dataclass(frozen=True)
class InstructionStats:
    counts: int
    counts_per_type: Mapping[str, int]

    all_instr_types: ClassVar[Set[str]] = {
        "arithmetic",
        "cast",
        "control",
        "logic",
        "memory",
    }

    @property
    def n_real_instructions(self) -> int:
        return sum(self.counts_per_type.values())

    @staticmethod
    def from_json(data: Mapping) -> "InstructionStats":
        counts_per_type = {
            instr_type: count for instr_type, count in data["type"].items()
        }
        if set(counts_per_type) != InstructionStats.all_instr_types:
            raise ValueError("Failed to parse statistics")

        return InstructionStats(
            data["count"],
            counts_per_type,
        )


@dataclass(frozen=True)
class IRStats:
    n_blocks: int
    instructions: InstructionStats

    @staticmethod
    def from_file(fp: str) -> "IRStats":
        with open(fp, "r") as f:
            # Let parsing errors propagate (i.e. ignore benchmarks for which we failed
            # to generate statistics)
            data: Mapping = json.load(f)
            return IRStats(
                data["basic-blocks"]["count"],
                InstructionStats.from_json(data["instructions"]),
            )


@dataclass(frozen=True)
class BenchStats:
    llvm: IRStats
    mlir: IRStats
    mlir_opt: IRStats

    @staticmethod
    def from_file(bench: str) -> "BenchStats":
        llvm = IRStats.from_file(_path_llvm(bench))
        mlir = IRStats.from_file(_path_mlir(bench))
        # TODO re-enable once we figure out polyhedral stuff
        # mlir_opt = IRStats.from_file(_path_mlir_opt(bench))
        return BenchStats(llvm, mlir, mlir)


def parse() -> Stats:
    benchmarks = sorted(os.listdir(_PATH))
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
    return stats


_PATH: Final[str] = os.path.join("benchmarks")


def _path_llvm(name: str) -> str:
    return os.path.join(_PATH, name, "llvm", "stats.json")


def _path_mlir(name: str) -> str:
    return os.path.join(_PATH, name, "mlir", "stats.json")


def _path_mlir_opt(name: str) -> str:
    return os.path.join(_PATH, name, "mlir", "stats_opt.json")
