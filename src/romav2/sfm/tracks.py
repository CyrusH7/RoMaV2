"""Remove verified match edges that make multi-view tracks self-conflicting."""

import numpy as np
from scipy.cluster.hierarchy import DisjointSet


def consistent_edge_mask(image_indices, node_a, node_b, scores):
    """Choose high-confidence edges whose components have one node per image.

    ``node_a`` and ``node_b`` index a shared array of node image indices. This
    runs on geometrically verified edges, so rejected raw matches cannot occupy
    a track. Stable sorting gives deterministic results when scores are equal.
    """
    image_indices = np.asarray(image_indices)
    node_a = np.asarray(node_a, dtype=np.int64)
    node_b = np.asarray(node_b, dtype=np.int64)
    scores = np.asarray(scores)
    if not (len(node_a) == len(node_b) == len(scores)):
        raise ValueError("Edge endpoints and scores must have equal lengths")
    parent = np.arange(len(image_indices), dtype=np.int64)
    # A root's image set is allocated only after it joins another node.
    members = {}
    kept = np.zeros(len(scores), dtype=bool)

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for edge in np.argsort(-scores, kind="stable"):
        a, b = root(int(node_a[edge])), root(int(node_b[edge]))
        if a == b:
            kept[edge] = True
            continue
        ia = members.get(a, {int(image_indices[a])})
        ib = members.get(b, {int(image_indices[b])})
        if not ia.isdisjoint(ib):
            continue
        if len(ia) < len(ib):
            a, b, ia, ib = b, a, ib, ia
        parent[b] = a
        ia.update(ib)
        members[a] = ia
        members.pop(b, None)
        kept[edge] = True
    return kept


def filter_verified_tracks(db, names, pairs, ids, store):
    """Replace verified geometries with confidence-ordered consistent edges."""
    offsets, image_indices, total = {}, [], 0
    for image_index, name in enumerate(names):
        count = len(store.keypoints(name))
        offsets[name] = total
        total += count
        image_indices.append(np.full(count, image_index, dtype=np.int32))
    image_indices = np.concatenate(image_indices)
    geometries, starts = [], [0]
    all_a, all_b, all_scores = [], [], []
    for a, b in pairs:
        geometry = db.read_two_view_geometry(ids[a], ids[b])
        matches = np.asarray(geometry.inlier_matches, dtype=np.uint32).reshape(-1, 2)
        raw = store.matches[a, b]
        scores = store.scores[a, b]
        # Recover the original confidence of each geometrically verified edge.
        raw_keys = (raw[:, 0].astype(np.uint64) << np.uint64(32)) | raw[:, 1]
        keys = (matches[:, 0].astype(np.uint64) << np.uint64(32)) | matches[:, 1]
        order = np.argsort(raw_keys)
        positions = np.searchsorted(raw_keys, keys, sorter=order)
        if len(keys) and (
            np.any(positions >= len(raw_keys))
            or np.any(raw_keys[order[positions]] != keys)
        ):
            raise ValueError(f"Verified matches missing from original pair {a}, {b}")
        all_a.append(matches[:, 0].astype(np.int64) + offsets[a])
        all_b.append(matches[:, 1].astype(np.int64) + offsets[b])
        all_scores.append(scores[order[positions]])
        geometries.append(geometry)
        starts.append(starts[-1] + len(matches))
    mask = consistent_edge_mask(
        image_indices,
        np.concatenate(all_a),
        np.concatenate(all_b),
        np.concatenate(all_scores),
    )
    db.clear_two_view_geometries()
    pair_counts = {}
    for i, ((a, b), geometry) in enumerate(zip(pairs, geometries)):
        geometry.inlier_matches = np.asarray(geometry.inlier_matches)[
            mask[starts[i] : starts[i + 1]]
        ]
        db.write_two_view_geometry(ids[a], ids[b], geometry)
        pair_counts[f"{a} {b}"] = len(geometry.inlier_matches)
    return pair_counts, int((~mask).sum())


def track_component_sizes(model):
    components = DisjointSet(model.reg_image_ids())
    for point in model.points3D.values():
        elements = point.track.elements
        for element in elements[1:]:
            components.merge(elements[0].image_id, element.image_id)
    return sorted((len(group) for group in components.subsets()), reverse=True)
