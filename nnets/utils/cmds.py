"""
Useful commands to copy/paste from/to the VM.

"""
import socket
from pathlib import Path


def download_cmd(rundir):
    """Commands to download files from VM and create local dirs.

    Downloads all CSV/YAML/LOG files from run_dir.

    """
    hostname = socket.gethostname()
    if 'dev' in rundir:
        idx = rundir.index("dev")
        dest = f"{rundir[idx:]}"
    else:
        dest = rundir
    print(
        f"\nDownload (VM -> local):\n\n"
        f"mkdir -p ~/{dest}; "
        f"gcloud compute scp "
        f"{hostname}:'{rundir}/*.[cyl][sao][vmg]*' "
        f"~/{dest} --zone us-central1-c "
    )


def evaluate_cmd(
    rundir,
    data="gs://scratch_fernando/infra_tiles_v1.zarr",
    split="test/base",
    home="/Users/fspaolo",
    bucket="scratch_fernando",
    module="classification.infra",
):
    """Commands to evaluate inference against test set (locally)."""
    idx = rundir.index("dev")
    dest = f"{rundir[idx:]}"
    if 'gs://' not in data:
        data = f"gs://{bucket}/{Path(data).name}"
    print(
        f"\nEvaluate (local):\n\n"
        f"python -m {module}.evaluate "
        f"--config-path ~/{dest} "
        f"data={data} split={split}\n"
    )
