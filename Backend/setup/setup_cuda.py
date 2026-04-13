from setup.setup_cuda import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy
    from pathlib import Path

    runpy.run_path(str(Path(__file__).resolve().parent / "setup" / "setup_cuda.py"), run_name="__main__")