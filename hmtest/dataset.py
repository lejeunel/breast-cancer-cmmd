import typer
from typing_extensions import Annotated
from pathlib import Path
from collections import namedtuple
import concurrent.futures

IMAGE_DOWNLOAD_URL = r"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={}"
BreastsScan = namedtuple("BreastsScan", ["left", "right"])


def fetch_one_patient(series_uid, out_dir, i=None, total=None) -> BreastsScan:
    """
    Fetch raw DICOM given series_uid from cancerimagingarchive API,
    and return results as a set of DICOM objects (one for each breast)
    """
    import zipfile
    import io
    import requests

    url = IMAGE_DOWNLOAD_URL.format(series_uid)
    response = requests.get(url)

    zip_bytes = response._content
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")
    for name in zf.namelist():
        if name.endswith(".dcm"):
            out_path = Path(out_dir) / "{}-{}".format(series_uid, name)
            dicom_bytes = zf.read(name)
            with open(out_path, "wb") as f:
                f.write(dicom_bytes)

    msg = ""

    if i is not None and total is not None:
        msg = "[{}/{}] ".format(i + 1, total)

    msg += "{}".format(series_uid)
    print(msg)


def fetch_raw_data(
    meta_path: Annotated[Path, typer.Argument(help="path to output unified csv file")],
    out_path: Annotated[
        Path,
        typer.Argument(
            help="output path where DICOM files are saved", exists=True, writable=True
        ),
    ],
    workers: Annotated[int, typer.Option("-w", help="Number of worker threads")] = 32,
):
    """
    Fetch raw DICOMs using provided meta-data and save to disk
    """
    import pandas as pd

    meta = pd.read_csv(meta_path)

    payloads = [
        [meta.iloc[i]["Series UID"], out_path, i, meta.shape[0]]
        for i in range(meta.shape[0])
    ]

    fetch_one_patient(*payloads[0])

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Start the load operations and mark each future with its URL
        future_to_scans = {executor.submit(fetch_one_patient, *p): p for p in payloads}
        for future in concurrent.futures.as_completed(future_to_scans):
            scan = future_to_scans[future]
            try:
                data = future.result()
            except Exception as exc:
                print("generated an exception: %s" % (exc))
