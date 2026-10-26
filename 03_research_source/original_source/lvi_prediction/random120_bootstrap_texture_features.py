"""T2/ADC habitats and standard non-shape PyRadiomics texture descriptors."""
import logging
import numpy as np
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import slic_habitat_backend as slic
from quantile_habitat_features import original_values

MODALITIES = ("T2", "ADC")
CLASSES = ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")


def whole_textures(profiles, indices, ids):
    radiomics.logger.setLevel(logging.ERROR)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=.25, normalize=False,
        label=1, distances=[1], additionalInfo=False, force2D=False)
    extractor.disableAllFeatures()
    for group in CLASSES: extractor.enableFeatureClassByName(group)
    rows = {"base": [], "erode": [], "dilate": []}
    metadata, audit = None, []
    for n, i in enumerate(indices, 1):
        current = {kind: [] for kind in rows}
        meta = []
        for mod in MODALITIES:
            p = profiles[int(i)][mod]
            image = sitk.GetImageFromArray(np.asarray(p["z"][..., 0], np.float32))
            image.SetSpacing(tuple(p["spacing"])[::-1])
            names = None
            for kind, maskkey in [("base", "roi"), ("erode", "erode"), ("dilate", "dilate")]:
                roi = np.asarray(p[maskkey], np.uint8)
                mask = sitk.GetImageFromArray(roi)
                mask.CopyInformation(image)
                try:
                    values = {key: float(value) for key, value in extractor.execute(image, mask).items()
                              if key.startswith("original_")}
                    error = None
                except ValueError as exc:
                    if kind == "base": raise
                    values, error = {}, str(exc)
                if names is None: names = sorted(values)
                assert len(names) == 93
                assert not any("_shape_" in name for name in names)
                current[kind].extend(values.get(name, np.nan) for name in names)
                if kind == "base":
                    meta.extend({"name": mod+"__Whole_"+name, "modality": mod, "block": "Whole"} for name in names)
                audit.append({"AnalysisID": ids[int(i)], "Modality": mod, "Variant": kind,
                    "FeatureCount": len(values), "Error": error})
        if metadata is None: metadata = meta
        assert meta == metadata
        for kind in rows: rows[kind].append(current[kind])
        if n % 10 == 0: print(f"T2/ADC texture extraction {n}/{len(indices)}", flush=True)
    return {**{kind: np.asarray(value, float) for kind, value in rows.items()}, "metadata": metadata,
        "indices": np.asarray(indices, int), "audit": audit, "binWidth": .25,
        "pyradiomics_version": radiomics.__version__, "feature_classes": CLASSES}


def habitat_values(p, mod, partition):
    labels = slic.assign(p, partition)
    roi, band = p["roi"], p["boundary"]
    z = p["z"][..., 0]
    channels = [("Z", z)] + ([("Raw", original_values(p, mod)[..., 0])] if mod == "ADC" else [])
    values, metadata = [], []
    fractions = {}

    def add(value, name):
        values.append(float(value))
        metadata.append({"name": mod+"__Boot_"+name, "modality": mod, "block": "Habitat"})

    for h in range(1, 4):
        region = roi & (labels == h)
        tumor_fraction = region.sum()/max(roi.sum(), 1)
        band_fraction = ((labels == h) & band).sum()/max(band.sum(), 1)
        fractions[h] = (tumor_fraction, band_fraction)
        add(tumor_fraction, f"H{h}_TumorFraction")
        add(band_fraction, f"H{h}_NativeBandFraction")
        for channel, image in channels:
            a = image[region]
            if len(a) >= 12:
                p10, q25, median, q75, p90 = np.quantile(a, [.1, .25, .5, .75, .9])
                stats = [p10, median, p90, q75-q25]
            else: stats = [np.nan]*4
            for val, name in zip(stats, ["P10", "Median", "P90", "IQR"]): add(val, f"H{h}_{channel}_{name}")
    for j, group in enumerate(["Tumor", "NativeBand"]):
        for h in [1, 2]: add(np.log((fractions[h][j]+.005)/(fractions[3][j]+.005)), f"LogH{h}H3_{group}")
    return np.asarray(values), metadata


def bootstrap_tables(profiles, train, evaluation, seed):
    # Uniform training-patient bootstraps; outcomes never enter partitions.
    train, evaluation = np.asarray(train, int), np.asarray(evaluation, int)
    rng = np.random.default_rng(seed)
    combined = np.r_[train, evaluation]
    measurements, metadata, fitted, membership = [], None, [], []
    for r in range(5):
        drawn = rng.choice(train, len(train), replace=True)
        partitions = {mod: slic.fit_partition(profiles, drawn, mod, 3, seed+100*r) for mod in MODALITIES}
        sample, meta = [], None
        for i in combined:
            row, names = [], []
            for mod in MODALITIES:
                values, current = habitat_values(profiles[int(i)][mod], mod, partitions[mod])
                row.extend(values); names.extend(current)
            if meta is None: meta = names
            assert names == meta
            sample.append(row)
        if metadata is None: metadata = meta
        assert metadata == meta
        measurements.append(sample); fitted.append(partitions); membership.append(drawn)
    measured = np.asarray(measurements, float).transpose(1, 0, 2)
    return {"train_measurements": measured[:len(train)], "evaluation_measurements": measured[len(train):],
        "metadata": metadata, "partitions": fitted, "bootstrap_train_indices": membership,
        "train": train, "evaluation": evaluation, "seed": seed}


def inference_habitats(profiles, indices, partitions):
    measured, metadata = [], None
    for fitted in partitions:
        rows = []
        for i in indices:
            row, names = [], []
            for mod in MODALITIES:
                value, meta = habitat_values(profiles[int(i)][mod], mod, fitted[mod])
                row.extend(value); names.extend(meta)
            if metadata is None: metadata = names
            assert metadata == names
            rows.append(row)
        measured.append(rows)
    return {"base": np.nanmean(np.asarray(measured, float), axis=0), "metadata": metadata}
