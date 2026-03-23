from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "_fieldnames"):
        return {name: getattr(obj, name) for name in obj._fieldnames}
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    raise TypeError(f"Unsupported MATLAB object type: {type(obj)!r}")


def _decode_if_bytes(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "U"}:
        if value.ndim == 0:
            return str(value.item())
        return "".join(str(x) for x in value.tolist())
    return value


def _load_eeglab_fdt(
    fdt_path: Path,
    nbchan: int,
    pnts: int,
    trials: int,
) -> np.ndarray:
    raw = np.fromfile(fdt_path, dtype=np.float32)
    expected = nbchan * pnts * trials
    if raw.size != expected:
        raise ValueError(
            f"Unexpected .fdt size for {fdt_path}. "
            f"Expected {expected} float32 values, found {raw.size}."
        )
    data = raw.reshape((nbchan, pnts, trials), order="F")
    if trials == 1:
        data = data[:, :, 0]
    return data


def _load_eeg_from_mat(path: Path) -> tuple[np.ndarray, float, list[str]]:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "EEG" in mat:
        eeg = _as_dict(mat["EEG"])
    elif "data" in mat and "srate" in mat:
        eeg = {key: value for key, value in mat.items() if not key.startswith("__")}
    else:
        raise ValueError(f"Missing EEG variable in {path}")
    data = eeg["data"]
    srate = float(np.asarray(eeg["srate"]).item())
    nbchan = int(np.asarray(eeg.get("nbchan", 0)).item())
    pnts = int(np.asarray(eeg.get("pnts", 0)).item())
    trials = int(np.asarray(eeg.get("trials", 1)).item())

    if isinstance(data, str) or isinstance(data, bytes) or (
        isinstance(data, np.ndarray) and data.dtype.kind in {"S", "U"}
    ):
        fdt_name = _decode_if_bytes(data)
        fdt_path = path.with_name(fdt_name)
        data = _load_eeglab_fdt(fdt_path, nbchan=nbchan, pnts=pnts, trials=trials)
    else:
        data = np.asarray(data, dtype=np.float32)
        if nbchan == 0:
            nbchan = int(data.shape[0])
        if pnts == 0:
            pnts = int(data.shape[-1])

    chanlocs = eeg.get("chanlocs", [])
    channel_names: list[str] = []
    if isinstance(chanlocs, np.ndarray):
        chanlocs = chanlocs.tolist()
    if not isinstance(chanlocs, list):
        chanlocs = [chanlocs]
    for item in chanlocs:
        try:
            channel_names.append(str(_as_dict(item).get("labels", f"ch{len(channel_names):02d}")))
        except TypeError:
            channel_names.append(f"ch{len(channel_names):02d}")

    if data.ndim == 3:
        data = data.reshape(data.shape[0], -1)
    if data.shape[0] != nbchan and data.shape[1] == nbchan:
        data = data.T

    return data.astype(np.float32), srate, channel_names


def _load_eeg_from_h5(path: Path) -> tuple[np.ndarray, float, list[str]]:
    with h5py.File(path, "r") as handle:
        if "EEG" not in handle:
            raise ValueError(f"Missing EEG group in {path}")
        eeg = handle["EEG"]
        data = eeg["data"]
        if isinstance(data, h5py.Dataset):
            arr = data[()]
        else:
            ref = data[()]
            arr = handle[ref]
        arr = np.asarray(arr)
        srate = float(np.asarray(eeg["srate"][()]).squeeze())
        if arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1, order="F")
        if arr.shape[0] > arr.shape[1]:
            arr = arr.T
        return arr.astype(np.float32), srate, []


def load_eeg(path: str | Path) -> tuple[np.ndarray, float, list[str]]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        data = np.load(file_path).astype(np.float32)
        return data, 200.0, [f"ch{i:02d}" for i in range(data.shape[0])]
    if suffix == ".npz":
        with np.load(file_path) as npz:
            if "data" not in npz:
                raise ValueError(f"Expected 'data' key in {file_path}")
            data = np.asarray(npz["data"], dtype=np.float32)
            fs = float(npz.get("srate", 200.0))
        return data, fs, [f"ch{i:02d}" for i in range(data.shape[0])]

    try:
        return _load_eeg_from_mat(file_path)
    except NotImplementedError:
        return _load_eeg_from_h5(file_path)


def _pick_first_array(values: dict[str, Any]) -> np.ndarray:
    for key in (
        "roi_time_series",
        "timeseries",
        "time_series",
        "data",
        "X",
        "arr",
        "roi",
    ):
        if key in values:
            return np.asarray(values[key], dtype=np.float32)
    for value in values.values():
        if isinstance(value, (np.ndarray, list, tuple, pd.DataFrame)):
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim in (1, 2):
                return arr
    raise ValueError("Could not find ROI array in object.")


def load_roi_timeseries(
    path: str | Path,
    n_rois: int = 64,
    return_names: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[str] | None]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    column_names: list[str] | None = None

    if suffix == ".pkl":
        with open(file_path, "rb") as handle:
            obj = pickle.load(handle)
    elif suffix == ".npy":
        obj = np.load(file_path)
    elif suffix == ".npz":
        with np.load(file_path) as npz:
            obj = {k: npz[k] for k in npz.files}
    elif suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        obj = pd.read_csv(file_path, sep=sep)
    else:
        raise ValueError(f"Unsupported ROI format: {file_path}")

    if isinstance(obj, pd.DataFrame):
        column_names = [str(col) for col in obj.columns.tolist()]
        arr = obj.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    elif isinstance(obj, dict):
        arr = _pick_first_array(obj)
    else:
        arr = np.asarray(obj, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr[:, None]

    if arr.shape[1] in {n_rois, n_rois + 2}:
        pass
    elif arr.shape[0] in {n_rois, n_rois + 2}:
        arr = arr.T
    elif arr.shape[1] <= max(n_rois + 2, 96) and arr.shape[0] > arr.shape[1]:
        pass
    elif arr.shape[0] <= max(n_rois + 2, 96) and arr.shape[1] > arr.shape[0]:
        arr = arr.T

    if arr.shape[1] > n_rois:
        arr = arr[:, :n_rois]
        if column_names is not None:
            column_names = column_names[:n_rois]

    arr = arr.astype(np.float32)
    if return_names:
        return arr, column_names
    return arr
