import subprocess
import sys
import os


def execute_pipeline():

    print("=" * 60)
    print("LAB 5 - PRODUCTION DATA PIPELINE")
    print("STROKE PREDICTION PROJECT")
    print("=" * 60)

    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    scripts = [
        "src/validate_data.py",
        "src/preprocess_pipeline.py",
        "src/validate_outputs.py"
    ]

    for script in scripts:

        script_path = os.path.join(project_root, script)

        print("\n" + "-" * 60)
        print(f"[INFO] Executing: {script}")
        print("-" * 60)

        result = subprocess.run(
            [sys.executable, script_path],
            cwd=project_root
        )

        if result.returncode != 0:

            print("\n[ERROR] Pipeline halted.")
            print(f"[ERROR] Failed script: {script}")

            sys.exit(1)

    print("\n" + "=" * 60)
    print("[SUCCESS] LAB 5 PRODUCTION PIPELINE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    execute_pipeline()
    