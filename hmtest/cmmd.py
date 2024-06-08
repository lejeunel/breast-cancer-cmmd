import concurrent.futures
from dataclasses import dataclass
from pathlib import Path

import typer
from tqdm import tqdm
from typing_extensions import Annotated

IMAGE_DOWNLOAD_URL = r"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={}"
DICOM_ORIENTATION_TAG = (0x0020, 0x0020)


@dataclass
class BreastImage:
    serie_id: str
    side: str
    orientation: str
    filename: str
    abnormality: str = ""
    classification: str = ""
    subtype: str = ""


def fetch_and_save_one_patient(
    series_uid: str, out_dir: Path, i: int = None, total: int = None
):
    """
    Fetch raw DICOM given series_uid from cancerimagingarchive API,
    and save results as DICOM files
    """
    import io
    import zipfile

    import requests

    url = IMAGE_DOWNLOAD_URL.format(series_uid)
    response = requests.get(url)

    zip_bytes = response._content
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")
    for name in zf.namelist():
        if name.endswith(".dcm"):
            out_path = Path(out_dir)
            zf.extract(name, out_path)

    zf.close()

    msg = ""

    if i is not None and total is not None:
        msg = "[{}/{}] ".format(i + 1, total)

    msg += "{}".format(series_uid)
    print(msg)


def fetch_raw_data(
    meta_path: Annotated[Path, typer.Argument(help="path to csv file containing UIDs")],
    out_path: Annotated[
        Path,
        typer.Argument(
            help="output path where DICOM files are saved", exists=True, writable=True
        ),
    ],
    workers: Annotated[int, typer.Option("-w", help="Number of worker threads")] = 32,
):
    """
    Fetch DICOM files from server using provided meta-data and save to disk
    """
    import pandas as pd

    meta = pd.read_csv(meta_path)

    payloads = [
        [r["serie_id"], out_path / r["serie_id"], i, meta.shape[0]]
        for i, r in meta.iterrows()
    ]

    print(f"meta file contains {len(payloads)} scans")

    payloads = [p for p in payloads if not p[1].exists()]
    print(f"{len(payloads)} download(s) remaining ")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_scans = {
            executor.submit(fetch_and_save_one_patient, *p): p for p in payloads
        }
        for future in concurrent.futures.as_completed(future_to_scans):
            scan = future_to_scans[future]
            try:
                data = future.result()
            except Exception as exc:
                print("Exception: %s" % (exc))


def merge_meta_and_annotations(meta: Path, annotations: Path, out: Path):
    """
    Builds a single meta-data file from list of series (one row per patient),
    and annotations (one row per label)
    """
    import pandas as pd

    meta = pd.read_csv(meta)
    annotations = pd.read_csv(annotations)
    merged = pd.merge(meta, annotations, left_on="Subject ID", right_on="ID1")
    merged.sort_values("ID1", inplace=True)

    merged.drop(columns="Subject ID", inplace=True)
    merged.rename(
        columns={
            "ID1": "patient_id",
            "Series UID": "serie_id",
            "Study ID": "study_id",
            "Number of Images": "num_images",
            "LeftRight": "left_right",
        },
        inplace=True,
    )

    merged.left_right = merged.left_right.replace({"L": "left", "R": "right"})
    merged.rename(columns={"left_right": "side"}, inplace=True)

    print(f"saving merge meta-data to {out}")
    merged.to_csv(out, index=False)


def _traverse_dicom_dirs(dicom_path: Path) -> list[BreastImage]:
    """
    Walk directory containing dicom files to extract side, angle, and file name for each file.
    Return a list of BreastImage.
    """
    from pydicom import dcmread

    images = []

    scans = [d for d in dicom_path.iterdir()]
    for d in tqdm(scans):
        for f in d.glob("*.dcm"):
            ds = dcmread(f)
            orientation = ds[*DICOM_ORIENTATION_TAG].value
            image = BreastImage(
                side="left" if orientation in [["A", "R"], ["A", "FR"]] else "right",
                orientation=(
                    "low" if orientation in [["P", "L"], ["P", "FL"]] else "high"
                ),
                filename=f.name,
                serie_id=d.name,
            )
            images.append(image)

    return images


def build_per_image_meta(meta: Path, dicom_root_path: Path, out: Path):
    """
    Traverse DICOM directory and retrieve meta-data (body-side, angle) on each image.
    Then, append corresponding meta-data (annotations, ...).

    """
    import pandas as pd

    if out.exists():
        print(f"found file at {out}, quitting")
        return

    print(f"parsing DICOM directory: {dicom_root_path}...")
    images = _traverse_dicom_dirs(dicom_root_path)

    meta = pd.read_csv(meta)
    meta.subtype = meta.subtype.fillna("")

    print("building meta-data file...")
    n_skipped = 0
    for im in tqdm(images):
        meta_ = meta.loc[(meta.serie_id == im.serie_id) & (meta.side == im.side)]
        if not meta_.empty:
            im.abnormality = meta_.iloc[0].abnormality
            im.classification = meta_.iloc[0].classification
            im.subtype = meta_.iloc[0].subtype
        else:
            n_skipped += 1

    if n_skipped > 0:
        print(f"[!!!] found {n_skipped} images without annotations")

    print(f"found {len(images)} images with annotations")
    meta_out = pd.DataFrame.from_records([im.__dict__ for im in images])
    print(f"writing meta-data to {out}")
    meta_out.to_csv(out, index=False)


def dicom_to_png(
    meta: Path,
    dicom_root_path: Path,
    out_root_path: Path,
    width: Annotated[int, typer.Option(help="Output width in pixel")] = 512,
):
    """
    Convert DICOM images to PNG.
    """
    import pandas as pd
    from pydicom import dcmread
    from skimage.io import imsave
    from skimage.transform import resize
    import numpy as np

    meta = pd.read_csv(meta)
    print(f"converting and saving images to {out_root_path}")
    for i in tqdm(range(len(meta))):
        r = meta.iloc[i]
        out_path = out_root_path / r.serie_id / (r.filename.split(".")[0] + ".png")
        if out_path.exists():
            continue

        out_path.parent.mkdir(exist_ok=True)
        image = dcmread(dicom_root_path / r.serie_id / r.filename).pixel_array
        height = int(image.shape[0] * (width / image.shape[1]))
        image = resize(image, (height, width), preserve_range=True, anti_aliasing=True)
        image = image.astype(np.uint8)

        imsave(out_path, image)


app = typer.Typer()
app.command(help="Fetch raw data")(fetch_raw_data)
app.command(help="Merge meta data")(merge_meta_and_annotations)
app.command(help="Append file names")(build_per_image_meta)
app.command(help="Convert DICOM images to PNG")(dicom_to_png)
