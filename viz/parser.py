import os
import json
from dataclasses import dataclass

from typing import ClassVar, Set, Mapping, Dict, Sequence

Stats = Mapping[str, "BenchStats"]


class ParsingError(Exception):
    pass


@dataclass(frozen=True)
class BasicBlockStats:
    counts: int
    pred_counts: Sequence[int]
    succ_counts: Sequence[int]

    @staticmethod
    def from_json(data: Mapping) -> "BasicBlockStats":
        return BasicBlockStats(data["count"], data["predCounts"], data["succCounts"])

    @staticmethod
    def empty() -> "BasicBlockStats":
        return BasicBlockStats(0, [], [])


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

    @staticmethod
    def empty() -> "InstructionStats":
        return InstructionStats(0, {ty: 0 for ty in InstructionStats.all_instr_types})


@dataclass(frozen=True)
class IRStats:
    basic_blocks: BasicBlockStats
    instructions: InstructionStats
    is_empty: bool

    @staticmethod
    def from_file(fp: str) -> "IRStats":
        with open(fp, "r") as f:
            # Let parsing errors propagate (i.e. ignore benchmarks for which we failed
            # to generate statistics)
            data: Mapping = json.load(f)
            return IRStats(
                BasicBlockStats.from_json(data["basic-blocks"]),
                InstructionStats.from_json(data["instructions"]),
                False,
            )

    @staticmethod
    def empty() -> "IRStats":
        return IRStats(BasicBlockStats.empty(), InstructionStats.empty(), True)


@dataclass(frozen=True)
class BenchStats:
    llvm: IRStats
    mlir: IRStats
    mlir_opt: IRStats

    @staticmethod
    def from_file(path: str, allow_empty: bool = True) -> "BenchStats":
        def parse_ir(ir_path: str, name: str) -> IRStats:
            ir = IRStats.empty()
            try:
                ir = IRStats.from_file(ir_path)
            except Exception as e:
                if not allow_empty:
                    raise ParsingError(f"[{name}] {e}")
            return ir

        llvm = parse_ir(_path_llvm(path), "LLVM")
        mlir = parse_ir(_path_mlir(path), "MLIR")
        mlir_opt = parse_ir(_path_mlir_opt(path), "MLIR opt")
        return BenchStats(llvm, mlir, mlir_opt)


def parse(path: str) -> Stats:
    print(f"---- Parsing test suite at {path} ----")

    benchmarks = sorted(os.listdir(path))
    print(f"Detected {len(benchmarks)} benchmarks")

    print(f"Parsing statistics")
    n_fail: int = 0
    stats: Dict[str, BenchStats] = {}
    for bench in benchmarks:
        try:
            stats[bench] = BenchStats.from_file(os.path.join(path, bench))
        except ParsingError as e:
            print(f"Failed to parse benchmark: {bench}\n\t-> {e}")
            n_fail += 1
    print(f"\t{n_fail}/{len(benchmarks)} benchmarks failed to parse")

    n_opt = sum(1 for s in stats.values() if not s.mlir_opt.is_empty)
    print(f"{n_opt}/{len(stats)} benchmarks were affine optimized successfully")

    print(f"---- Done parsing ----\n")
    return stats


def _path_llvm(path: str) -> str:
    return os.path.join(path, "llvm", "stats.json")


def _path_mlir(path: str) -> str:
    return os.path.join(path, "mlir", "stats.json")


def _path_mlir_opt(path: str) -> str:
    return os.path.join(path, "mlir", "stats_opt.json")
