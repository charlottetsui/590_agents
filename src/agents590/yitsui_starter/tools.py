import asyncio
from pathlib import Path


async def run_command(command: str, cwd: Path) -> str:
    """Run a shell command and return its combined stdout and stderr."""
    process = await asyncio.create_subprocess_shell(
        command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output, _ = await process.communicate()
    return output.decode(errors="replace")
