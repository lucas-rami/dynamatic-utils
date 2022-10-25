# Libraries
import os
import shutil


def main() -> None:
    # Output directory
    out_dir = os.path.join("benchmarks", "graphs")

    # Clean up output directory
    shutil.rmtree(out_dir, ignore_errors=True)

    # Parse all benchmarks
    dynamatic_dir = os.path.join("benchmarks", "dynamatic")
    benchmarks = sorted(os.listdir(dynamatic_dir))

    n_transfer: int = 0
    for bench in benchmarks:
        # Get images
        llvm = os.path.join(dynamatic_dir, bench, "llvm", f"{bench}.png")
        mlir = os.path.join(dynamatic_dir, bench, "mlir", f"{bench}.png")

        if not os.path.isfile(llvm) or not os.path.isfile(mlir):
            print(f"Missing graph for {bench}")
            continue

        # Move them to destination folder
        dst = os.path.join(out_dir, bench)
        os.makedirs(dst, exist_ok=True)
        shutil.copyfile(llvm, os.path.join(dst, "llvm.png"))
        shutil.copyfile(mlir, os.path.join(dst, "mlir.png"))
        n_transfer += 1

    print(f"Number of graphs transfered: {n_transfer}")


if __name__ == "__main__":
    main()
