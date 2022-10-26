# Libraries
import os
import subprocess


def main() -> None:

    # Parse all benchmarks
    dynamatic_dir = os.path.join("benchmarks", "polybench")
    benchmarks = sorted(os.listdir(dynamatic_dir))

    for bench in benchmarks:
        affine = os.path.join(dynamatic_dir, bench, "mlir", "affine_fun.mlir")
        affine_opt = os.path.join(dynamatic_dir, bench, "mlir", "affine_opt_fun.mlir")

        if not os.path.isfile(affine) or not os.path.isfile(affine_opt):
            print(f"Missing IR for {bench}")
            continue

        # Diff command
        diff = subprocess.run(
            f'diff "{affine}" "{affine_opt}"',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
        )

        print(f"Diff for {bench}:\n{diff.stdout}")


if __name__ == "__main__":
    main()
