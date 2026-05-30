# 1. Installation

## 1.1 Recommended method

The recommended way is to use via `uv` ([https://docs.astral.sh/uv]()).

If you only want to run the command-line interface (CLI): after installing `uv`, this should be as simple as `uvx D95eq` or `uv tool install D95eq`.

If you want to import `D95eq` in some Python code, once you are within a `uv` project (`uv init`), you can install the module with `uv add D95eq`.

After installation, open a new shell window and try `D95eq --help`.

## 1.2 Other methods

You can of course install globally via `pip` (`pip install D95eq`), or only install the CLI using `pipx` (`pipx install D95eq`).
