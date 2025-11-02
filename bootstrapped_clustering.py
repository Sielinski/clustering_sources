import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

# --- Head/Tail breaks on a 1-D array ---
def head_tail_breaks(x, min_head_frac=0.4, max_iter=50):
    """
    Returns monotonically increasing breakpoints [min, m1, m2, ..., max].
    Stops when head is not 'small enough' or no further split is possible.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.array([])
    cur = x.copy()
    breaks = [np.nanmin(x)]
    it = 0
    while it < max_iter:
        m = cur.mean()
        head = cur[cur > m]
        if head.size == 0 or head.size / cur.size > min_head_frac:
            breaks.append(np.nanmax(cur))
            break
        breaks.append(m)
        cur = head
        it += 1
    # ensure strictly increasing and unique (numerical safety)
    b = np.unique(np.array(breaks, dtype=float))
    if b.size == 1:  # all the same value
        b = np.array([b[0], b[0]])
    return b

# --- Bootstrap head/tail on an already-aggregated vector (1-D) ---
def bootstrap_htb_values(
    values: np.ndarray | pd.Series,
    n_boot: int = 1000,
    confidence_level: float = 0.95,
    min_head_frac: float = 0.4,
    random_state: int | None = None,
):
    """
    Bootstraps the head/tail breaks on a 1-D array of values.
    Outputs:
      - break_ci: CI for each breakpoint index across bootstraps
      - memberships: per-item tier probabilities (+ MAP tier)
      - tiers_per_boot: distribution of #tiers observed across bootstraps
    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(values, dtype=float)
    n = len(x)

    # store breakpoints of each bootstrap (variable length)
    boots_breaks = []
    # membership tallies: rows = items, cols = up to max #tiers across boots
    max_bins_seen = 0
    # we'll collect membership tallies lazily once we know max bins
    all_labels = []

    for _ in range(n_boot):
        samp = rng.choice(x, size=n, replace=True)
        # compute breaks on this boot's counts
        b = head_tail_breaks(vals, min_head_frac=min_head_frac)
        if b.size < 2:
            continue
        boots_breaks.append(b)
        max_bins_seen = max(max_bins_seen, b.size - 1)
        # assign original items to tiers using this bootstrap's breaks
        lbl = pd.cut(x, bins=b, include_lowest=True, labels=False)
        all_labels.append(lbl.to_numpy())

    if len(boots_breaks) == 0:
        raise ValueError("No valid bootstrap runs produced head/tail breaks.")

    # --- Tier membership probabilities for each item ---
    memberships = np.zeros((n, max_bins_seen), dtype=float)
    valid = 0
    for b, lbl in zip(boots_breaks, all_labels):
        k = b.size - 1
        valid += 1
        for t in range(k):
            memberships[:, t] += (lbl == t)

    memberships = memberships / np.maximum(valid, 1)
    map_tier = memberships.argmax(axis=1)
    max_prob = memberships.max(axis=1)

    memberships_df = pd.DataFrame(memberships, columns=[f"p_tier_{i}" for i in range(max_bins_seen)])
    memberships_df["map_tier"] = map_tier
    memberships_df["max_prob"] = max_prob

    # --- Breakpoint CIs (by breakpoint index) ---
    # collect the j-th internal break across boots that have it
    # index j=0 is the LOWER bound (min), j=last is the UPPER bound (max) — usually not that informative,
    # so we report internal breaks (1..len(b)-2)
    alpha = 1 - confidence_level
    lo_q, hi_q = alpha/2, 1 - alpha/2
    max_breaks = max(b.size for b in boots_breaks)  # number of points, bins = points-1
    rows = []
    for j in range(max_breaks):
        vals = [b[j] for b in boots_breaks if b.size > j]
        if len(vals) == 0:
            continue
        rows.append({
            "break_index": j,
            "mean": float(np.mean(vals)),
            "ci_low": float(np.quantile(vals, lo_q)),
            "ci_high": float(np.quantile(vals, hi_q)),
            "n_boots_used": len(vals)
        })
    break_ci = pd.DataFrame(rows)

    # distribution of number of tiers over bootstraps
    tiers_per_boot = pd.Series([b.size - 1 for b in boots_breaks], name="n_tiers").value_counts().sort_index()

    return {
        "break_ci": break_ci,
        "memberships": memberships_df,
        "tiers_dist": tiers_per_boot
    }

# --- Bootstrap head/tail when you must resample by unit and re-aggregate counts ---
def bootstrap_htb_counts_by_unit(
    df: pd.DataFrame,
    group_col: str,          # e.g. "domain" or "url"
    unit_col: str,           # e.g. "response_id" (sampling unit)
    n_boot: int = 1000,
    confidence_level: float = 0.95,
    min_head_frac: float = 0.4,
    random_state: int | None = None,
):
    """
    Each bootstrap:
      1) resample units (with replacement),
      2) keep rows whose unit is selected,
      3) recompute counts per group,
      4) compute head/tail breaks and tier labels for the ORIGINAL groups.
    Returns:
      - memberships: per-group tier probabilities (+ MAP)
      - break_ci: bootstrap CIs for breakpoints
      - tiers_dist: distribution of #tiers across bootstraps
    """
    rng = np.random.default_rng(random_state)

    # fixed list of target groups (so output is aligned)
    groups = df[group_col].dropna().unique()
    groups = np.asarray(groups)
    G = len(groups)
    idx = {g: i for i, g in enumerate(groups)}

    units = df[unit_col].dropna().unique()
    units = np.asarray(units)
    if units.size == 0:
        raise ValueError("No resampling units found.")

    # membership tallies will grow to max tiers seen
    memberships = None
    boots_breaks = []
    valid_boots = 0

    for _ in range(n_boot):
        samp_units = rng.choice(units, size=units.size, replace=True)
        sub = df[df[unit_col].isin(samp_units)]

        # recompute counts per group in this bootstrap
        cnt = sub.groupby(group_col).size()
        # align to full group list (missing groups get 0)
        vals = np.zeros(G, dtype=float)
        ix = [idx[g] for g in cnt.index if g in idx]
        vals[ix] = cnt.values.astype(float)

        # compute breaks on this boot's counts
        b = head_tail_breaks(vals, min_head_frac=min_head_frac)
        if b.size < 2:
            continue

        k = b.size - 1
        # integer tier labels 0..k-1 without NaNs
        labels = np.digitize(vals, b[1:-1], right=True)

        # expand membership matrix if we saw more tiers than before
        if memberships is None:
            memberships = np.zeros((G, k), dtype=float)
        elif k > memberships.shape[1]:
            memberships = np.pad(memberships, ((0,0),(0, k - memberships.shape[1])), mode="constant", constant_values=0.0)

        for t in range(k):
            memberships[:, t] += (labels == t)

        boots_breaks.append(b)
        valid_boots += 1

    if valid_boots == 0:
        raise ValueError("No valid bootstrap runs produced head/tail breaks.")

    # probabilities + MAP
    memberships = memberships / valid_boots
    max_prob = memberships.max(axis=1)
    map_tier = memberships.argmax(axis=1)

    memberships_df = (pd.DataFrame(memberships, columns=[f"p_tier_{i}" for i in range(memberships.shape[1])])
                      .assign(**{group_col: groups, "map_tier": map_tier, "max_prob": max_prob})
                      .loc[:, [group_col, "map_tier", "max_prob"] + [c for c in memberships.shape[1] and [f"p_tier_{i}" for i in range(memberships.shape[1])]]])

    # breakpoint CIs
    alpha = 1 - confidence_level
    lo_q, hi_q = alpha/2, 1 - alpha/2
    max_breaks = max(b.size for b in boots_breaks)
    rows = []
    for j in range(max_breaks):
        vals = [b[j] for b in boots_breaks if b.size > j]
        if not vals:
            continue
        rows.append({
            "break_index": j,
            "mean": float(np.mean(vals)),
            "ci_low": float(np.quantile(vals, lo_q)),
            "ci_high": float(np.quantile(vals, hi_q)),
            "n_boots_used": len(vals)
        })
    break_ci = pd.DataFrame(rows)
    tiers_dist = pd.Series([b.size - 1 for b in boots_breaks], name="n_tiers").value_counts().sort_index()

    return {
        "memberships": memberships_df.sort_values(["map_tier", "max_prob"], ascending=[True, False]).reset_index(drop=True),
        "break_ci": break_ci,
        "tiers_dist": tiers_dist
    }


# ---------- utilities ----------

def _elbow_k(ks, inertias):
    x = np.asarray(ks, float); y = np.asarray(inertias, float)
    x0, y0 = x[0], y[0]; x1, y1 = x[-1], y[-1]
    denom = np.hypot(x1 - x0, y1 - y0)
    if denom == 0: 
        return 2 if len(ks) >= 2 else ks[0]
    d = np.abs((y1 - y0) * x - (x1 - x0) * y + (x1 * y0 - y1 * x0)) / denom
    return int(max(2, x[np.argmax(d)]))

def _kmeans_inertia_curve(X, k_min=1, k_max=10, n_init=20, random_state=0):
    ks, inertias = [], []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        km.fit(X)
        ks.append(k); inertias.append(km.inertia_)
    return ks, inertias

def _align_labels(ref_centers, cur_centers):
    """Align current k-means labels to reference via Hungarian assignment."""
    from scipy.spatial.distance import cdist
    D = cdist(cur_centers, ref_centers)
    row_ind, col_ind = linear_sum_assignment(D)
    perm = np.empty(len(cur_centers), dtype=int)
    perm[row_ind] = col_ind
    return perm

# ---------- feature construction ----------

def build_group_features(
    df: pd.DataFrame,
    group_col: str,
    unit_col: str,
    extra_feature_cols: list[str] | None = None,
    count_unique_by_unit: bool = True,
):
    """
    Returns [group_col, citation_count, initial_response, ...extra_feature_cols]

    extra_feature_cols:
      list of numeric columns (e.g., one-hots) to aggregate per group.
    count_unique_by_unit:
      If True, for extra features we first aggregate to max per (group, unit)
      so a unit only contributes once per feature, then sum across units.
      This avoids double-counting repeated rows within the same unit.
    """
    needed = [group_col, unit_col]
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in input df.")

    # Base features
    counts = df.groupby(group_col).size().rename("citation_count")
    if "initial_response" in df.columns:
        init = df.groupby(group_col)["initial_response"].min()
    else:
        d2 = df.reset_index(drop=True).reset_index(names="row_ix")
        init = d2.groupby(group_col)["row_ix"].min().rename("initial_response")

    out = pd.concat([counts, init], axis=1).reset_index(names=group_col)

    # Extra features (e.g., one-hot aspect columns)
    if extra_feature_cols:
        missing_extras = [c for c in extra_feature_cols if c not in df.columns]
        if missing_extras:
            raise KeyError(f"Missing extra_feature_cols in input df: {missing_extras}")

        # Coerce extras to numeric (protect against object dtypes)
        dfx = df[[group_col, unit_col] + extra_feature_cols].copy()
        for c in extra_feature_cols:
            dfx[c] = pd.to_numeric(dfx[c], errors="coerce").fillna(0)

        if count_unique_by_unit:
            # one-hot per (group, unit): take max within unit, then sum over units
            tmp = (
                dfx.groupby([group_col, unit_col])[extra_feature_cols]
                   .max()
                   .groupby(group_col)[extra_feature_cols]
                   .sum()
                   .reset_index()
            )
        else:
            tmp = dfx.groupby(group_col)[extra_feature_cols].sum().reset_index()

        out = out.merge(tmp, on=group_col, how="left").fillna(0)

    return out

# ---------- main: bootstrap + k-means ----------

def bootstrap_kmeans_groups(
    citations_df: pd.DataFrame,
    group_col: str,                 # "domain" or "url"
    unit_col: str,                  # "query_id" or "response_id"
    feature_cols=("citation_count", "initial_response"),
    k_min: int = 1,
    k_max: int = 10,
    n_init: int = 20,
    n_boot: int = 300,
    confidence_level: float = 0.95,
    random_state: int | None = 0,
):
    """
    2-pass k-means with bootstrap uncertainty.

    Parameters
    ----------
    feature_cols : str or Sequence[str]
        Names of columns in the per-group table to use as features.
        Defaults to ("citation_count", "initial_response").
        You may pass a single column, e.g., feature_cols="citation_count".

    Returns
    -------
    dict with keys: k, memberships, centers, scree, ref_centers
    """
    # --- normalize feature_cols to a list ---
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]
    else:
        feature_cols = list(feature_cols)

    rng = np.random.default_rng(random_state)

    # Split requested features into "base" (computed here) vs "extra" (exist in raw df)
    BASE_FEATURES = {"citation_count", "initial_response"}
    extra_cols = [c for c in feature_cols if c not in BASE_FEATURES]

    # ---- pooled features (deterministic) ----
    pooled_stats = build_group_features(
        citations_df, group_col, unit_col,
        extra_feature_cols=extra_cols,   # <-- only extras are pulled from raw df
        count_unique_by_unit=True
    )

    # ensure requested feature columns exist
    missing = [c for c in feature_cols if c not in pooled_stats.columns]
    if missing:
        raise KeyError(f"Missing feature column(s) in pooled stats: {missing}")

    X_raw = pooled_stats[feature_cols].to_numpy(float)
    n_samples = X_raw.shape[0]
    if n_samples < 2:
        raise ValueError("Need at least 2 groups to cluster.")

    # standardize
    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)

    # ---- select k on pooled data ----
    k_max = min(k_max, n_samples)
    ks, inertias = _kmeans_inertia_curve(X, k_min=k_min, k_max=k_max, n_init=n_init, random_state=0)
    k_selected = _elbow_k(ks, inertias)
    k_selected = min(k_selected, n_samples)
    scree_df = pd.DataFrame({"k": ks, "inertia": inertias})

    # ---- reference fit (for label alignment) ----
    km_ref = KMeans(n_clusters=k_selected, n_init=n_init, random_state=0).fit(X)
    ref_centers_scaled = km_ref.cluster_centers_
    ref_centers = scaler.inverse_transform(ref_centers_scaled)

    # ---- prep bootstrap resampling of units ----
    if unit_col not in citations_df.columns:
        raise KeyError(f"Missing unit_col '{unit_col}' in citations_df.")
    units = citations_df[unit_col].dropna().unique()
    units = np.asarray(units)
    M = len(units)
    if M == 0:
        raise ValueError("No resampling units found.")

    # collectors
    G = len(pooled_stats)
    k = k_selected
    assign_counts = np.zeros((G, k), dtype=int)
    centers_boot = np.full((n_boot, k, len(feature_cols)), np.nan, dtype=float)
    idx_map = {g: i for i, g in enumerate(pooled_stats[group_col].to_list())}

    # ---- bootstrap loop ----
    for b in range(n_boot):
        samp_units = rng.choice(units, size=M, replace=True)
        sub = citations_df[citations_df[unit_col].isin(samp_units)]

        bs_stats = build_group_features(
            sub, group_col, unit_col,
            extra_feature_cols=extra_cols,   # <-- only extras here too
            count_unique_by_unit=True
        )

        # keep only entities present in pooled set
        bs_stats = bs_stats[bs_stats[group_col].isin(idx_map.keys())].copy()
        if bs_stats.empty:
            continue

        # ensure features exist
        if any(c not in bs_stats.columns for c in feature_cols):
            # if a custom feature is missing in this bootstrap, skip it
            continue

        Xb_raw = bs_stats[feature_cols].to_numpy(float)
        Xb = scaler.transform(Xb_raw)

        if False:
            # keep the randomness in k-means controlled by rng
            seed = int(rng.integers(0, 2**31 - 1))
            km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(Xb)
        else:
            # alternativley, a fresh seed for each fit
            km = KMeans(n_clusters=k, n_init=n_init, random_state=None).fit(Xb)


        # align labels
        perm = _align_labels(ref_centers_scaled, km.cluster_centers_)
        aligned_labels = np.take(perm, km.labels_)
        aligned_centers_scaled = km.cluster_centers_[np.argsort(perm)]
        aligned_centers = scaler.inverse_transform(aligned_centers_scaled)

        # update assignment tallies
        for g_name, lab in zip(bs_stats[group_col].to_numpy(), aligned_labels):
            assign_counts[idx_map[g_name], lab] += 1

        centers_boot[b, :, :] = aligned_centers

    # ---- membership probabilities & MAP labels ----
    probs = assign_counts / np.maximum(assign_counts.sum(axis=1, keepdims=True), 1)
    map_labels = probs.argmax(axis=1)
    max_prob = probs.max(axis=1)

    memberships = pooled_stats[[group_col]].copy()
    memberships["cluster"] = map_labels
    memberships["max_prob"] = max_prob
    for c in range(k):
        memberships[f"p_cluster_{c}"] = probs[:, c]

    # ---- cluster centers with CIs (original units) ----
    alpha = 1.0 - confidence_level
    lo_q, hi_q = alpha/2.0, 1.0 - alpha/2.0
    center_rows = []
    valid_mask = ~(np.isnan(centers_boot).all(axis=(1,2)))
    Cboot = centers_boot[valid_mask]
    if Cboot.size > 0:
        for c in range(k):
            for j, feat in enumerate(feature_cols):
                vals = Cboot[:, c, j]
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    continue
                center_rows.append({
                    "cluster": c,
                    "feature": feat,
                    "mean": float(np.nanmean(vals)),
                    "ci_low": float(np.nanquantile(vals, lo_q)),
                    "ci_high": float(np.nanquantile(vals, hi_q)),
                })
    centers_ci = pd.DataFrame(center_rows)

    return {
        "k": k_selected,
        "memberships": memberships.sort_values(["cluster", "max_prob"], ascending=[True, False]).reset_index(drop=True),
        "centers": centers_ci,
        "scree": scree_df,
        "ref_centers": pd.DataFrame(ref_centers, columns=feature_cols).assign(cluster=lambda d: np.arange(k)),
    }