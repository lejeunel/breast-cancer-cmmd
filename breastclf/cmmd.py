import concurrent.futures
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import typer
from pydicom import dcmread
from skimage.io import imsave
from skimage.transform import resize
from tqdm import tqdm
from typing_extensions import Annotated

IMAGE_DOWNLOAD_URL = r"https://services.cancerimagingarchive.net/services/v4/TCIA/query/getImage?SeriesInstanceUID={}"
DICOM_PATIENT_ID_TAG = (0x0010, 0x0020)
DICOM_IMG_LATERALITY_TAG = (0x0020, 0x0062)
AMBIGUOUS_IMAGES = [
    ("D1-0202", "00000001.dcm"),
    ("D2-0284", "00000001.dcm"),
    ("D1-0202", "00000002.dcm"),
    ("D2-0284", "00000002.dcm"),
    ("D1-0202", "00000003.dcm"),
    ("D2-0284", "00000003.dcm"),
    ("D1-0202", "00000004.dcm"),
    ("D2-0284", "00000004.dcm"),
    ("D1-0808", "00000001.dcm"),
    ("D1-1292", "00000001.dcm"),
]


@dataclass
class BreastImage:
    serie_id: str
    patient_id: str
    side: str
    filename: str
    abnormality: str = ""
    classification: str = ""
    subtype: str = ""


def _fetch_and_save_one_patient(
    series_uid: str, out_dir: Path, i: int = None, total: int = None
):
    """
    Fetch zip archive containing raw DICOM serie given series_uid from
    cancerimagingarchive API and save results locally
    """

    url = IMAGE_DOWNLOAD_URL.format(series_uid)
    response = requests.get(url)
    # print(f"response: {response}")

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
    n_workers: Annotated[int, typer.Option("-w", help="Number of worker threads")] = 32,
):
    """
    Fetch DICOM files from server using provided meta-data and save to disk
    """

    meta = pd.read_csv(meta_path)

    assert (
        "serie_id" in meta.columns
    ), "provided meta-data file must contain necessary column serie_id"

    payloads = [
        [r["serie_id"], out_path / r["serie_id"], i, meta.shape[0]]
        for i, r in meta.iterrows()
    ]

    print(f"found {len(payloads)} series")

    payloads = [p for p in payloads if not p[1].exists()]
    print(f"{len(payloads)} download(s) remaining ")

    if len(payloads) == 0:
        return

    print(f"fetching files with {n_workers} threads")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_scans = {
            executor.submit(_fetch_and_save_one_patient, *p): p for p in payloads
        }
        for future in concurrent.futures.as_completed(future_to_scans):
            future_to_scans[future]
            try:
                future.result()
            except Exception as exc:
                print("Exception: %s" % (exc))


def merge_meta_and_annotations(meta: Path, annotations: Path, out: Path):
    """
    Builds a single meta-data file from list of series (one row per patient),
    and annotations (one row per label)
    """

    if out.exists():
        print(f"found file {out}, we are done.")
        return

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
            "LeftRight": "side",
        },
        inplace=True,
    )

    merged.side = merged.side.replace({"L": "left", "R": "right"})

    print(f"saving merged meta-data to {out}")
    merged.to_csv(out, index=False)


def _traverse_dicom_dirs(dicom_path: Path) -> list[BreastImage]:
    """
    Walk directory containing dicom files to extract the following relevant
    meta-data: side, orientation, and file name for each file.
    """

    images = []

    scans = [d for d in dicom_path.iterdir()]
    for d in tqdm(scans):
        for f in d.glob("*.dcm"):
            ds = dcmread(f)
            laterality = ds[DICOM_IMG_LATERALITY_TAG].value
            patient_id = ds[DICOM_PATIENT_ID_TAG].value
            image = BreastImage(
                side="left" if laterality == "L" else "right",
                patient_id=patient_id,
                filename=f.name,
                serie_id=d.name,
            )
            images.append(image)

    return images


def _remove_ambiguous_images(meta: pd.DataFrame) -> pd.DataFrame:
    """
    based on the dataset repository
    the hashes for the pixels of the following seem to be identical.
    TCIA does not know which is the “more correct” case for the files mentioned:
    D1-0202 (1-1.dcm image) and D2-0284 (1-1.dcm image)
    D1-0202 (1-2.dcm image) and D2-0284 (1-2.dcm image)
    D1-0202 (1-3.dcm image) and D2-0284 (1-3.dcm image)
    D1-0202 (1-4.dcm image) and D2-0284 (1-4.dcm image)
    D1-0808 (1-1.dcm image) and D1-1292 (1-1.dcm image)

    We therefore remove those images from the meta-data file
    """
    ambiguous_cols = (
        meta[["patient_id", "filename"]].apply(tuple, axis=1).isin(AMBIGUOUS_IMAGES)
    )
    meta = meta.loc[~ambiguous_cols].reset_index(drop=True)

    return meta


def build_per_image_meta(meta: Path, dicom_root_path: Path, out: Path):
    """
    Traverse DICOM directory and retrieve meta-data (body-side, angle) on each image.
    Then, append corresponding meta-data (annotations, ...).

    """
    if out.exists():
        print(f"found file {out}, we are done.")
        return

    meta = pd.read_csv(meta)
    meta.subtype = meta.subtype.fillna("")

    print(f"parsing DICOM directory: {dicom_root_path}...")
    images = _traverse_dicom_dirs(dicom_root_path)

    print("building meta-data file...")
    n_skipped = 0
    for im in tqdm(images):
        meta_ = meta.loc[(meta.serie_id == im.serie_id) & (meta.side == im.side)]
        if not meta_.empty:
            im.abnormality = meta_.iloc[0].abnormality
            im.classification = meta_.iloc[0].classification
            im.subtype = meta_.iloc[0].subtype
            im.patient_id = meta_.iloc[0].patient_id
        else:
            n_skipped += 1

    if n_skipped > 0:
        print(f"[!!!] found {n_skipped} images without annotations")

    print(f"found {len(images)} images with annotations")
    meta_out = pd.DataFrame.from_records([im.__dict__ for im in images])
    print(f"Removing ambiguous images {AMBIGUOUS_IMAGES}")
    meta_out = _remove_ambiguous_images(meta_out)
    print(f"writing meta-data to {out}")
    meta_out.to_csv(out, index=False)


def dicom_to_png(
    meta: Path,
    dicom_root_path: Path,
    out_root_path: Annotated[Path, typer.Argument(help="output path", exists=True)],
    width: Annotated[int, typer.Option(help="Output width in pixel")] = 2048,
):
    """
    Convert DICOM images to PNG.
    """

    meta = pd.read_csv(meta)
    out_paths = [
        out_root_path / r.serie_id / (r.filename.split(".")[0] + ".png")
        for _, r in meta.iterrows()
    ]
    remaining = [p.exists() == False for p in out_paths]
    print(f"Found {sum(remaining)} remaining files out of {len(out_paths)} to convert")

    if sum(remaining) == 0:
        return

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
