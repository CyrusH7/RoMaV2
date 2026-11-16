"""Import externally computed correspondences and run COLMAP incremental SfM."""
from pathlib import Path
import json
import logging

import numpy as np
from PIL import Image
import pycolmap

from .matching import FeatureStore
from .tracks import filter_verified_tracks

LOG = logging.getLogger(__name__)


def reconstruct(images: Path, output: Path, names, pairs, matcher, *, cell_size=1.0,
                camera_params=None, min_matches=15, fix_intrinsics=False, seed=0):
    images, output = Path(images).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output directory must be empty: {output}')
    if len(set(names)) != len(names) or any(any(c.isspace() for c in n) for n in names):
        raise ValueError('Image names must be unique and cannot contain whitespace')
    if min_matches < 8:
        raise ValueError('min_matches must be at least 8')
    if fix_intrinsics and camera_params is None:
        raise ValueError('--fix-intrinsics requires known --camera-params')
    sizes = {}
    for name in names:
        path = (images / name).resolve()
        if not path.is_relative_to(images) or not path.is_file():
            raise ValueError(f'Image path is outside image directory or missing: {name}')
        with Image.open(path) as im:
            sizes[name] = im.size
    if camera_params is not None:
        camera_params = np.asarray(camera_params, dtype=float)
        if (camera_params.shape != (4,) or not np.isfinite(camera_params).all()
                or np.any(camera_params[:2] <= 0)):
            raise ValueError('PINHOLE parameters must be finite fx fy cx cy with positive focal lengths')
        if len(set(sizes.values())) != 1:
            raise ValueError('A shared calibrated camera requires images of identical dimensions')
    names_set = set(names)
    if not pairs or any(a not in names_set or b not in names_set or a == b for a, b in pairs):
        raise ValueError('Pairs must reference distinct selected images')

    output.mkdir(parents=True, exist_ok=True)
    store = FeatureStore(cell_size)
    for index, (a, b) in enumerate(pairs, 1):
        LOG.info('Matching %d/%d: %s <-> %s', index, len(pairs), a, b)
        xy_a, xy_b, scores = matcher(images / a, images / b)
        store.add(a, b, xy_a, xy_b, scores, sizes)
    db_path, pairs_path = output / 'database.db', output / 'pairs.txt'
    pairs_path.write_text(''.join(f'{a} {b}\n' for a, b in pairs), encoding='utf-8')
    reader = pycolmap.ImageReaderOptions()
    if camera_params is not None:
        reader.camera_model = 'PINHOLE'
        reader.camera_params = ','.join(map(str, camera_params))
        mode = pycolmap.CameraMode.SINGLE
    else:
        mode = pycolmap.CameraMode.AUTO
    pycolmap.Database.open(str(db_path)).close()
    pycolmap.import_images(str(db_path), str(images), camera_mode=mode,
                           image_names=list(names), options=reader)
    db = pycolmap.Database.open(str(db_path))
    try:
        ids = {}
        for name in names:
            image = db.read_image_with_name(name)
            if image is None:
                raise RuntimeError(f'COLMAP failed to import image: {name}')
            ids[name] = image.image_id
            db.write_keypoints(image.image_id, store.keypoints(name))
        for (a, b), matches in store.matches.items():
            db.write_matches(ids[a], ids[b], matches)
    finally:
        db.close()
    geom_options = pycolmap.TwoViewGeometryOptions()
    geom_options.min_num_inliers = min_matches
    geom_options.ransac.random_seed = seed
    pycolmap.verify_matches(str(db_path), str(pairs_path), geom_options)
    db = pycolmap.Database.open(str(db_path))
    try:
        pair_counts, rejected_conflicts = filter_verified_tracks(db, names, pairs, ids, store)
    finally:
        db.close()
    # Dense pairwise matches can be large; release the Python copy before mapping.
    del store
    sparse = output / 'sparse'
    sparse.mkdir()
    options = pycolmap.IncrementalPipelineOptions()
    options.min_num_matches = min_matches
    options.min_model_size = 2
    options.random_seed = seed
    if fix_intrinsics:
        options.ba_refine_focal_length = False
        options.ba_refine_principal_point = False
        options.ba_refine_extra_params = False
        options.mapper.abs_pose_refine_focal_length = False
        options.mapper.abs_pose_refine_extra_params = False
    models = pycolmap.incremental_mapping(str(db_path), str(images), str(sparse), options=options)
    info = {'images': len(names), 'pairs': len(pairs),
            'verified_pairs': sum(v >= min_matches for v in pair_counts.values()),
            'verified_inliers': pair_counts, 'track_conflicts_rejected': rejected_conflicts, 'models': []}
    for model_id, model in sorted(models.items()):
        info['models'].append({'id': int(model_id), 'registered_images': model.num_reg_images(),
                               'points3D': model.num_points3D()})
    (output / 'summary.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
    if not models:
        raise RuntimeError(f'COLMAP did not reconstruct a model; see {output}/summary.json')
    return info
