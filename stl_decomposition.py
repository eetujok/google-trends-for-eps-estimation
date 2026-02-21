"""
STL Decomposition, Event Formation, Event Table Formation, and Earnings Report Merge

This module provides functionality for:
1. STL decomposition of time series data
2. Event detection using calendar-adjusted residuals
3. Event table formation with features
4. Merging events with earnings reports

PARALLELIZATION:
- CPU parallelization is supported via multiprocessing (n_jobs parameter)
- Each keyword's STL decomposition runs independently and can be parallelized
- Set n_jobs=None to use all CPU cores, n_jobs=1 for sequential processing

GPU ACCELERATION:
- STL decomposition from statsmodels does not support GPU acceleration
- For GPU acceleration, you would need to:
  1. Use GPU-accelerated libraries like CuPy or RAPIDS cuDF for data processing
  2. Implement or use GPU-accelerated STL decomposition (e.g., cuSignal)
  3. This would require significant code changes and additional dependencies
- Current CPU parallelization provides significant speedup for most use cases
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from statsmodels.tsa.seasonal import STL
import time
import sys
from multiprocessing import Pool, cpu_count
from functools import partial


def save_event_table(events_df, name=None, output_dir=None):
    """
    Save event table to output/event-table. Creates folder if needed.
    
    Args:
        events_df: DataFrame with event data
        name: Optional filename stem (default: timestamp)
        output_dir: Base output directory (default: ../output)
    
    Returns:
        Path to saved CSV file
    """
    if output_dir is None:
        output_dir = Path(".") / "data"
    else:
        output_dir = Path(output_dir)
    
    out_dir = output_dir / "event-table"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = name if name else f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path = out_dir / f"{stem}.csv"
    events_df.to_csv(path, index=False)
    return path


def stl_decomposition(
    ts,
    period=53,
    seasonal=53,
    trend=103
):
    """
    Perform STL decomposition on a time series and compute standardized residuals.
    
    Args:
        ts: pandas Series with datetime index
        period: Period for seasonality (default: 53)
        seasonal: Seasonal smoothing parameter (default: 53)
        trend: Trend smoothing parameter (default: 103)
    
    Returns:
        dict with keys: 'resid' (z_t), 'r_t' (standardized residual), 'r_cal_t' (calendar-adjusted residual)
    """
    # Ensure weekly frequency
    if ts.index.freq is None:
        ts = ts.resample("W-SAT").mean()
        ts = ts.ffill().bfill()
        ts.index.freq = "W-SAT"
    
    # STL decomposition
    stl = STL(ts, seasonal=seasonal, trend=trend, period=period)
    res = stl.fit()
    z_t = res.resid
    
    # Standardize residual to robust z-score
    m = z_t.median()
    mad = (z_t - m).abs().median()
    sigma_r = 1.4826 * mad
    r_t = (z_t - m) / sigma_r if sigma_r > 0 else (z_t - m)
    
    # Calendar-adjusted residual (residual-of-residuals): condition on week-of-year
    week = r_t.index.isocalendar().week
    by_week = r_t.groupby(week)
    mu_w = by_week.transform("median")  # μ_w = median(r_t | week(t)=w)
    mad_w = by_week.transform(lambda x: (x - x.median()).abs().median())  # MAD(r_t | week(t)=w)
    sigma_w = (1.4826 * mad_w).replace(0, float("nan")).fillna(1)  # avoid div by 0
    r_cal_t = (r_t - mu_w) / sigma_w  # r̃_t
    
    return {
        'resid': z_t,
        'r_t': r_t,
        'r_cal_t': r_cal_t
    }


def detect_events(
    r_cal_t,
    event_threshold_high=3.0,
    event_stop_pos=1.5,
    start_consec=1,
    end_consec=2
):
    """
    Detect event windows from calendar-adjusted residual using hysteresis.
    
    Args:
        r_cal_t: pandas Series with calendar-adjusted residual
        event_threshold_high: Start positive event when r_cal_t >= this (default: 3.0)
        event_stop_pos: Stop positive event after r_cal_t <= this (default: 1.5)
        start_consec: Require this many consecutive points to start an event (default: 1)
        end_consec: Require this many consecutive points past stop threshold to end an event (default: 2)
    
    Returns:
        List of (start_timestamp, end_timestamp) tuples
    """
    idx = r_cal_t.index
    x = r_cal_t.values
    
    upper_windows = []  # list of (start_ts, end_ts); only positive windows
    
    in_pos = False
    pos_start_i = None
    pos_above_start = 0
    pos_below_stop = 0
    
    for i in range(len(x)):
        if not in_pos:
            if x[i] >= event_threshold_high:
                pos_above_start += 1
                if pos_above_start >= start_consec:
                    in_pos = True
                    pos_start_i = i - start_consec + 1
                    pos_below_stop = 0
            else:
                pos_above_start = 0
        else:
            # continue while above stop; end after END_CONSEC points at/below stop
            if x[i] <= event_stop_pos:
                pos_below_stop += 1
                if pos_below_stop >= end_consec:
                    pos_end_i = i - end_consec
                    if pos_end_i >= pos_start_i:
                        upper_windows.append((idx[pos_start_i], idx[pos_end_i]))
                    in_pos = False
                    pos_start_i = None
                    pos_above_start = 0
                    pos_below_stop = 0
            else:
                pos_below_stop = 0
    
    if in_pos and pos_start_i is not None:
        upper_windows.append((idx[pos_start_i], idx[-1]))
    
    return upper_windows


def compute_event_features(
    t_s,
    t_e,
    ev,
    ev_cal,
    r_t,
    pre=8,
    post=8,
    tail=4,
    shift_thr=0.5
):
    """
    Compute features for a single event.
    
    Args:
        t_s: Event start timestamp
        t_e: Event end timestamp
        ev: pandas Series of r_t values during event window
        ev_cal: pandas Series of r_cal_t values during event window
        r_t: Full pandas Series of standardized residuals
        pre: Weeks before start for baseline (default: 8)
        post: Weeks after end for baseline (default: 8)
        tail: Weeks after peak for retention (default: 4)
        shift_thr: r-units threshold for regime_shift_flag_e (default: 0.5)
    
    Returns:
        dict with event features
    """
    t_p = ev.idxmax()
    peak_z_e = ev.max()
    area_z_e = ev.clip(lower=0).sum()
    
    duration_e = len(ev)
    seasonal_surprise_e = ev_cal.loc[t_p]
    
    pos_peak = ev.index.get_loc(t_p)
    weeks_to_peak = pos_peak
    weeks_from_peak_to_end = len(ev) - 1 - pos_peak
    
    # half_life_e (right-side): first time after peak where ev <= half-height
    h = 0.5 * peak_z_e
    after_peak = ev.loc[t_p:]
    crosses = after_peak <= h
    if crosses.any():
        t_half = crosses.idxmax()  # first True
        half_life_e = ev.index.get_loc(t_half) - pos_peak
    else:
        half_life_e = weeks_from_peak_to_end
    
    rise_slope_e = (peak_z_e - ev.iloc[0]) / max(1, weeks_to_peak)
    decay_slope_e = (ev.iloc[-1] - peak_z_e) / max(1, weeks_from_peak_to_end)
    asymmetry_e = (weeks_to_peak - weeks_from_peak_to_end) / max(1, duration_e)
    
    pre_slice = r_t.loc[(r_t.index >= t_s - pd.DateOffset(weeks=pre)) & (r_t.index < t_s)]
    post_slice = r_t.loc[(r_t.index > t_e) & (r_t.index <= t_e + pd.DateOffset(weeks=post))]
    pre_level = pre_slice.median() if len(pre_slice) > 0 else float("nan")
    post_level = post_slice.median() if len(post_slice) > 0 else float("nan")
    baseline_shift_z_e = (post_level - pre_level) if pd.notna(pre_level) and pd.notna(post_level) else float("nan")
    
    tail_slice = r_t.loc[(r_t.index > t_p) & (r_t.index <= t_p + pd.DateOffset(weeks=tail))]
    tail = tail_slice.mean() if len(tail_slice) > 0 else float("nan")
    retention_ratio_e = (tail / (peak_z_e + 1e-9)) if pd.notna(tail) and (abs(peak_z_e) + 1e-9) != 0 else float("nan")
    
    regime_shift_flag_e = 1 if (pd.notna(baseline_shift_z_e) and baseline_shift_z_e >= shift_thr) else 0
    
    return {
        "t_start": t_s,
        "t_peak": t_p,
        "t_end": t_e,
        "peak_z_e": peak_z_e,
        "area_z_e": area_z_e,
        "duration_e": duration_e,
        "seasonal_surprise_e": seasonal_surprise_e,
        "half_life_e": half_life_e,
        "rise_slope_e": rise_slope_e,
        "decay_slope_e": decay_slope_e,
        "asymmetry_e": asymmetry_e,
        "baseline_shift_z_e": baseline_shift_z_e,
        "retention_ratio_e": retention_ratio_e,
        "regime_shift_flag_e": regime_shift_flag_e,
    }


def process_keyword_events(
    keyword,
    ts,
    period=53,
    seasonal=53,
    trend=103,
    event_threshold_high=3.0,
    event_stop_pos=1.5,
    start_consec=1,
    end_consec=2,
    pre=8,
    post=8,
    tail=4,
    shift_thr=0.5,
    require_min_change=False,
    min_change_fraction=0.10,
    pre_weeks_for_avg=4,
    verbose=False
):
    """
    Process a single keyword: STL decomposition, event detection, and feature computation.
    
    Args:
        keyword: Keyword name
        ts: pandas DataFrame with 'date' and 'value' columns
        period: Period for seasonality (default: 53)
        seasonal: Seasonal smoothing parameter (default: 53)
        trend: Trend smoothing parameter (default: 103)
        event_threshold_high: Start positive event when r_cal_t >= this (default: 3.0)
        event_stop_pos: Stop positive event after r_cal_t <= this (default: 1.5)
        start_consec: Require this many consecutive points to start an event (default: 1)
        end_consec: Require this many consecutive points past stop threshold to end an event (default: 2)
        pre: Weeks before start for baseline (default: 8)
        post: Weeks after end for baseline (default: 8)
        tail: Weeks after peak for retention (default: 4)
        shift_thr: r-units threshold for regime_shift_flag_e (default: 0.5)
        require_min_change: If True, only create event if at least one value in the event
            window differs from the pre-event average by min_change_fraction (default: False)
        min_change_fraction: Minimum relative change vs 4-week pre-average required when
            require_min_change is True (default: 0.10 = 10%)
        pre_weeks_for_avg: Number of weeks before event start used to compute baseline
            average (default: 4)
        verbose: Whether to print debug info (default: False)
    
    Returns:
        List of event feature dictionaries
    """
    try:
        # Prepare time series
        ts = ts.copy()
        ts["date"] = pd.to_datetime(ts["date"], utc=True)
        ts = ts.sort_values("date").set_index("date").squeeze()
        if ts.empty:
            if verbose:
                print(f"  [process_keyword_events] '{keyword}': Empty time series")
            return []
        
        ts = ts.astype(float)
        ts = ts.resample("W-SAT").mean()
        ts = ts.ffill().bfill()
        if ts.empty or len(ts) < 2:
            if verbose:
                print(f"  [process_keyword_events] '{keyword}': Insufficient data after resampling")
            return []
        ts.index.freq = "W-SAT"
        
        if verbose:
            print(f"  [process_keyword_events] '{keyword}': Running STL decomposition on {len(ts)} points")
        
        # STL decomposition
        stl_start = time.time()
        stl_result = stl_decomposition(ts, period=period, seasonal=seasonal, trend=trend)
        if verbose:
            print(f"  [process_keyword_events] '{keyword}': STL completed in {time.time() - stl_start:.2f}s")
        
        r_t = stl_result['r_t']
        r_cal_t = stl_result['r_cal_t']
        
        # Detect events
        if verbose:
            print(f"  [process_keyword_events] '{keyword}': Detecting events...")
        upper_windows = detect_events(
            r_cal_t,
            event_threshold_high=event_threshold_high,
            event_stop_pos=event_stop_pos,
            start_consec=start_consec,
            end_consec=end_consec
        )
        
        if verbose:
            print(f"  [process_keyword_events] '{keyword}': Found {len(upper_windows)} event windows")
        
        # Compute features for each event
        event_rows = []
        for t_s, t_e in upper_windows:
            ev = r_t.loc[t_s:t_e]
            ev_cal = r_cal_t.loc[t_s:t_e]
            if ev.empty:
                continue

            if require_min_change:
                pre_slice = ts.loc[(ts.index >= t_s - pd.DateOffset(weeks=pre_weeks_for_avg)) & (ts.index < t_s)]
                pre_avg = pre_slice.mean()
                ev_slice = ts.loc[t_s:t_e]
                if len(ev_slice) == 0:
                    continue
                denom = max(abs(pre_avg), 1e-9) if pd.notna(pre_avg) else 1e-9
                rel_changes = (ev_slice - pre_avg).abs() / denom
                if not (rel_changes >= min_change_fraction).any():
                    continue

            features = compute_event_features(
                t_s, t_e, ev, ev_cal, r_t,
                pre=pre, post=post, tail=tail, shift_thr=shift_thr
            )
            features["keyword"] = keyword
            event_rows.append(features)
        
        return event_rows
    except Exception as e:
        if verbose:
            print(f"  [process_keyword_events] ERROR in '{keyword}': {e}")
        return []


def _process_single_keyword(args):
    """
    Helper function for multiprocessing: process a single keyword.
    This function must be at module level for pickling.
    """
    (kw, ts_dict, period, seasonal, trend, event_threshold_high,
     event_stop_pos, start_consec, end_consec, pre, post, tail, shift_thr,
     require_min_change, min_change_fraction, pre_weeks_for_avg) = args
    
    try:
        # Convert dict back to DataFrame
        ts = pd.DataFrame(ts_dict)
        ts["date"] = pd.to_datetime(ts["date"], utc=True)
        
        events = process_keyword_events(
            kw,
            ts,
            period=period,
            seasonal=seasonal,
            trend=trend,
            event_threshold_high=event_threshold_high,
            event_stop_pos=event_stop_pos,
            start_consec=start_consec,
            end_consec=end_consec,
            pre=pre,
            post=post,
            tail=tail,
            shift_thr=shift_thr,
            require_min_change=require_min_change,
            min_change_fraction=min_change_fraction,
            pre_weeks_for_avg=pre_weeks_for_avg,
            verbose=False
        )
        return (kw, events, None)
    except Exception as e:
        return (kw, [], str(e))


def build_event_table(
    df_all,
    min_nonzero_frac=1.0,
    period=53,
    seasonal=53,
    trend=103,
    event_threshold_high=3.0,
    event_stop_pos=1.5,
    start_consec=1,
    end_consec=2,
    pre=8,
    post=8,
    tail=4,
    shift_thr=0.5,
    require_min_change=False,
    min_change_fraction=0.10,
    pre_weeks_for_avg=4,
    output_dir=None,
    save=True,
    filename=None,
    verbose=True,
    n_jobs=None
):
    """
    Build event table for all keywords in the dataset.
    
    Args:
        df_all: DataFrame with columns ['keyword', 'date', 'value']
        min_nonzero_frac: Minimum fraction of non-zero values required (default: 1.0)
        period: Period for seasonality (default: 53)
        seasonal: Seasonal smoothing parameter (default: 53)
        trend: Trend smoothing parameter (default: 103)
        event_threshold_high: Start positive event when r_cal_t >= this (default: 3.0)
        event_stop_pos: Stop positive event after r_cal_t <= this (default: 1.5)
        start_consec: Require this many consecutive points to start an event (default: 1)
        end_consec: Require this many consecutive points past stop threshold to end an event (default: 2)
        pre: Weeks before start for baseline (default: 8)
        post: Weeks after end for baseline (default: 8)
        tail: Weeks after peak for retention (default: 4)
        shift_thr: r-units threshold for regime_shift_flag_e (default: 0.5)
        require_min_change: If True, only create event if at least one value in the event
            window differs from the pre-event average by min_change_fraction (default: False)
        min_change_fraction: Minimum relative change vs pre-event average when
            require_min_change is True (default: 0.10 = 10%)
        pre_weeks_for_avg: Number of weeks before event start for baseline average (default: 4)
        output_dir: Base output directory (default: ../output)
        save: Whether to save the event table (default: True)
        filename: Optional filename for saved table (default: None, uses timestamp)
        verbose: Whether to print progress (default: True)
        n_jobs: Number of parallel jobs (default: None = use all CPUs, 1 = sequential)
    
    Returns:
        DataFrame with event table
    """
    start_time = time.time()
    if verbose:
        print(f"[BUILD_EVENT_TABLE] Starting at {datetime.now().strftime('%H:%M:%S')}")
        print(f"[BUILD_EVENT_TABLE] Input: {len(df_all)} rows, {df_all['keyword'].nunique()} unique keywords")
    
    # Single groupby: reuse for filtering and for per-keyword data (avoids O(keywords*rows) repeated .loc)
    grouped = df_all.groupby("keyword", sort=False)
    if verbose:
        print(f"[BUILD_EVENT_TABLE] Filtering keywords by min_nonzero_frac >= {min_nonzero_frac}")
    keyword_nonzero_frac = grouped["value"].agg(lambda x: (x > 0).mean())
    keywords_all = [kw for kw in sorted(grouped.groups) if keyword_nonzero_frac[kw] >= min_nonzero_frac]
    
    # Test mode: limit number of keywords (for debugging)
    test_limit = getattr(build_event_table, '_test_limit', None)
    if test_limit and test_limit > 0:
        original_count = len(keywords_all)
        keywords_all = keywords_all[:test_limit]
        if verbose:
            print(f"[BUILD_EVENT_TABLE] TEST MODE: Limited to first {test_limit} keywords (from {original_count})")
    
    if verbose:
        print(f"[BUILD_EVENT_TABLE] After filtering: {len(keywords_all)} keywords to process")
    
    # Determine number of jobs
    if n_jobs is None:
        n_jobs = cpu_count()
    elif n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs < 1:
        n_jobs = 1
    
    if verbose:
        print(f"[BUILD_EVENT_TABLE] Using {n_jobs} parallel workers")
    
    # Prepare data for each keyword (O(1) per keyword via grouped.get_group; no full-df scan per kw)
    # Convert to dicts for pickling in multiprocessing
    keyword_data = []
    for kw in keywords_all:
        grp = grouped.get_group(kw)[["date", "value"]]
        ts_dict = grp.to_dict("records") if len(grp) > 0 else []
        keyword_data.append((
            kw, ts_dict, period, seasonal, trend, event_threshold_high,
            event_stop_pos, start_consec, end_consec, pre, post, tail, shift_thr,
            require_min_change, min_change_fraction, pre_weeks_for_avg
        ))
    
    event_rows_all = []
    errors = []
    
    if n_jobs == 1:
        # Sequential processing (easier debugging)
        if verbose:
            print(f"[BUILD_EVENT_TABLE] Processing sequentially...")
        processed = 0
        last_log_time = time.time()
        
        for args in keyword_data:
            # For sequential mode, convert dict back to DataFrame first
            (kw, ts_dict, period, seasonal, trend, event_threshold_high,
             event_stop_pos, start_consec, end_consec, pre, post, tail, shift_thr,
             require_min_change, min_change_fraction, pre_weeks_for_avg) = args
            ts = pd.DataFrame(ts_dict) if ts_dict else pd.DataFrame(columns=['date', 'value'])
            
            try:
                events = process_keyword_events(
                    kw,
                    ts,
                    period=period,
                    seasonal=seasonal,
                    trend=trend,
                    event_threshold_high=event_threshold_high,
                    event_stop_pos=event_stop_pos,
                    start_consec=start_consec,
                    end_consec=end_consec,
                    pre=pre,
                    post=post,
                    tail=tail,
                    shift_thr=shift_thr,
                    require_min_change=require_min_change,
                    min_change_fraction=min_change_fraction,
                    pre_weeks_for_avg=pre_weeks_for_avg,
                    verbose=False
                )
                event_rows_all.extend(events)
            except Exception as e:
                errors.append((kw, str(e)))
                if verbose:
                    print(f"[BUILD_EVENT_TABLE] ERROR processing keyword '{kw}': {e}")
            
            processed += 1
            # Progress logging every 10 keywords or every 30 seconds
            current_time = time.time()
            if verbose and (processed % 10 == 0 or (current_time - last_log_time) > 30):
                elapsed = current_time - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = (len(keywords_all) - processed) / rate if rate > 0 else 0
                print(f"[BUILD_EVENT_TABLE] Progress: {processed}/{len(keywords_all)} keywords "
                      f"({processed/len(keywords_all)*100:.1f}%), "
                      f"{len(event_rows_all)} events found, "
                      f"ETA: {remaining/60:.1f} min")
                last_log_time = current_time
                sys.stdout.flush()
    else:
        # Parallel processing
        if verbose:
            print(f"[BUILD_EVENT_TABLE] Processing in parallel with {n_jobs} workers...")
        
        with Pool(processes=n_jobs) as pool:
            # Use imap for progress tracking
            results = pool.imap(_process_single_keyword, keyword_data)
            processed = 0
            last_log_time = time.time()
            
            for kw, events, error in results:
                if error:
                    errors.append((kw, error))
                    if verbose:
                        print(f"[BUILD_EVENT_TABLE] ERROR processing keyword '{kw}': {error}")
                else:
                    event_rows_all.extend(events)
                
                processed += 1
                # Progress logging every 10 keywords or every 30 seconds
                current_time = time.time()
                if verbose and (processed % 10 == 0 or (current_time - last_log_time) > 30):
                    elapsed = current_time - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = (len(keywords_all) - processed) / rate if rate > 0 else 0
                    print(f"[BUILD_EVENT_TABLE] Progress: {processed}/{len(keywords_all)} keywords "
                          f"({processed/len(keywords_all)*100:.1f}%), "
                          f"{len(event_rows_all)} events found, "
                          f"ETA: {remaining/60:.1f} min")
                    last_log_time = current_time
                    sys.stdout.flush()
    
    if verbose and errors:
        print(f"[BUILD_EVENT_TABLE] Encountered {len(errors)} errors during processing")
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"[BUILD_EVENT_TABLE] Completed in {elapsed/60:.2f} minutes")
        print(f"[BUILD_EVENT_TABLE] Total events found: {len(event_rows_all)}")
    
    events_df = pd.DataFrame(event_rows_all)
    
    if save and len(events_df) > 0:
        if verbose:
            print(f"[BUILD_EVENT_TABLE] Saving event table...")
        event_table_path = save_event_table(events_df, name=filename, output_dir=output_dir)
        if verbose:
            print(f"[BUILD_EVENT_TABLE] Event table saved to {event_table_path}")
    
    return events_df


def merge_earnings_reports(
    events_df,
    output_dir=None,
    enrichment_dir=None,
    save=True,
    filename="quarter_events",
    verbose=True
):
    """
    Match keywords to companies and assign events to quarterly report intervals.
    
    Args:
        events_df: DataFrame with event table (must have 'keyword', 't_start', 't_end' columns)
        output_dir: Base output directory for metadata files (default: ../output)
        enrichment_dir: Directory for earnings data (default: ../enrichment)
        save: Whether to save the merged table (default: True)
        filename: Filename for saved table (default: "quarter_events")
        verbose: Whether to print progress (default: True)
    
    Returns:
        DataFrame with events merged with earnings reports
    """
    start_time = time.time()
    if verbose:
        print(f"[MERGE_EARNINGS] Starting at {datetime.now().strftime('%H:%M:%S')}")
        print(f"[MERGE_EARNINGS] Input: {len(events_df)} events")
    
    if output_dir is None:
        output_dir = Path(".") / "data"
    else:
        output_dir = Path(output_dir)
    
    if enrichment_dir is None:
        enrichment_dir = Path(".") / "data"
    else:
        enrichment_dir = Path(enrichment_dir)
    
    # 1. Load metadata: keyword -> list of companies (one keyword can have multiple companies)
    if verbose:
        print(f"[MERGE_EARNINGS] Loading metadata files from {output_dir}")
    meta_files = sorted(output_dir.glob("normalized_metadata_branded_*.csv"))
    if not meta_files:
        raise FileNotFoundError("No normalized_metadata_branded_*.csv found in output/")
    
    if verbose:
        print(f"[MERGE_EARNINGS] Found {len(meta_files)} metadata files")
    
    # Some metadata CSVs have malformed lines (extra commas); skip bad lines to avoid ParserError
    meta = pd.concat(
        [pd.read_csv(f, on_bad_lines="skip") for f in meta_files],
        ignore_index=True,
    )
    
    if verbose:
        print(f"[MERGE_EARNINGS] Loaded {len(meta)} metadata rows")
    
    # Drop duplicates so each (keyword, company) appears once; keep status=success if you have it
    if "status" in meta.columns:
        meta = meta.loc[meta["status"] == "success", ["keyword", "company"]]
    else:
        meta = meta[["keyword", "company"]]
    meta = meta.drop_duplicates()
    keyword_to_companies = meta.groupby("keyword")["company"].apply(list).to_dict()
    
    if verbose:
        print(f"[MERGE_EARNINGS] {len(keyword_to_companies)} keywords mapped to companies")
    
    # 2. Load earnings and build quarter intervals per company
    if verbose:
        print(f"[MERGE_EARNINGS] Loading earnings data from {enrichment_dir / 'earnings_data.csv'}")
    earnings = pd.read_csv(enrichment_dir / "earnings_data.csv")
    earnings["date"] = pd.to_datetime(earnings["date"], utc=True)
    earnings["reportDate"] = pd.to_datetime(earnings["reportDate"], utc=True)
    # Quarter: 3 months ending on `date`. Interval [quarter_start, quarter_end]
    earnings["quarter_end"] = earnings["date"]
    earnings["quarter_start"] = earnings["quarter_end"] - pd.DateOffset(months=3) + pd.Timedelta(days=1)
    
    if verbose:
        print(f"[MERGE_EARNINGS] Loaded {len(earnings)} earnings records")
    
    # Normalize company name for matching (strip + lower for case-insensitive join)
    def norm(s):
        return str(s).strip().lower()
    
    meta_companies = set(meta["company"].map(norm))
    earnings["company_norm"] = earnings["companyName"].map(norm)
    # Restrict earnings to companies that appear in metadata
    earnings = earnings.loc[earnings["company_norm"].isin(meta_companies)].copy()
    # Map metadata company (original) to one canonical form for join
    meta["company_norm"] = meta["company"].map(norm)
    keyword_to_companies_norm = meta.groupby("keyword")["company_norm"].apply(list).to_dict()
    
    if verbose:
        print(f"[MERGE_EARNINGS] {len(earnings)} earnings records match metadata companies")
        print(f"[MERGE_EARNINGS] Matching events to quarters...")
    
    # 3. Assign event rows to quarters: use the quarter that contains the event peak (t_peak).
    #    This ensures events that span a quarter boundary (e.g. spike late Q4 into early Q1) are
    #    still assigned to one quarter. Fall back to "whole event in quarter" if t_peak is missing.
    events = events_df.copy()
    events["t_start"] = pd.to_datetime(events["t_start"], utc=True)
    events["t_end"] = pd.to_datetime(events["t_end"], utc=True)
    if "t_peak" in events.columns:
        events["t_peak"] = pd.to_datetime(events["t_peak"], utc=True)
    
    quarter_event_rows = []
    processed = 0
    last_log_time = time.time()
    
    for idx, ev in events.iterrows():
        kw = ev["keyword"]
        companies = keyword_to_companies_norm.get(kw, [])
        if not companies:
            continue
        t_s, t_e = ev["t_start"], ev["t_end"]
        t_peak = ev.get("t_peak") if "t_peak" in ev.index else None
        for _c in companies:
            eq = earnings.loc[earnings["company_norm"] == _c]
            for _, q in eq.iterrows():
                q_start, q_end = q["quarter_start"], q["quarter_end"]
                # Prefer: quarter contains the event peak; else require whole event inside quarter
                if t_peak is not None and pd.notna(t_peak):
                    in_quarter = q_start <= t_peak <= q_end
                else:
                    in_quarter = t_s >= q_start and t_e <= q_end
                if in_quarter:
                    row = ev.to_dict()
                    row["company"] = q["companyName"]
                    row["company_norm"] = _c
                    row["quarter_start"] = q_start
                    row["quarter_end"] = q_end
                    row["reportDate"] = q["reportDate"]
                    row["ticker"] = q["ticker"]
                    row["exchange"] = q["exchange"]
                    row["epsActual"] = q["epsActual"]
                    row["epsEstimate"] = q["epsEstimate"]
                    row["epsDifference"] = q["epsDifference"]
                    row["surprisePercent"] = q["surprisePercent"]
                    row["currency"] = q["currency"]
                    quarter_event_rows.append(row)
                    break  # at most one quarter per company per event
        
        processed += 1
        # Progress logging every 1000 events or every 30 seconds
        current_time = time.time()
        if verbose and (processed % 1000 == 0 or (current_time - last_log_time) > 30):
            elapsed = current_time - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (len(events) - processed) / rate if rate > 0 else 0
            print(f"[MERGE_EARNINGS] Progress: {processed}/{len(events)} events "
                  f"({processed/len(events)*100:.1f}%), "
                  f"{len(quarter_event_rows)} matches found, "
                  f"ETA: {remaining/60:.1f} min")
            last_log_time = current_time
            sys.stdout.flush()
    
    quarter_events_df = pd.DataFrame(quarter_event_rows)
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"[MERGE_EARNINGS] Completed in {elapsed:.2f} seconds")
        print(f"[MERGE_EARNINGS] Events in event table: {len(events_df)}; Quarter/event rows: {len(quarter_events_df)}")
    
    if save and len(quarter_events_df) > 0:
        if verbose:
            print(f"[MERGE_EARNINGS] Saving quarter events table...")
        save_event_table(quarter_events_df, name=filename, output_dir=output_dir)
        if verbose:
            print(f"[MERGE_EARNINGS] Quarter events table saved")
    
    return quarter_events_df


def run_full_pipeline(
    output_dir=None,
    enrichment_dir=None,
    min_nonzero_frac=0.9,
    period=53,
    seasonal=53,
    trend=103,
    event_threshold_high=4.0,
    event_stop_pos=1.5,
    start_consec=1,
    end_consec=2,
    pre=8,
    post=8,
    tail=4,
    shift_thr=0.5,
    require_min_change=False,
    min_change_fraction=0.10,
    pre_weeks_for_avg=4,
    save_events=True,
    save_quarter_events=True,
    events_filename=None,
    quarter_events_filename="quarter_events",
    verbose=True,
    test_keyword_limit=None,
    n_jobs=None
):
    """
    Run the full pipeline: load data, build event table, and merge with earnings.
    
    Args:
        output_dir: Base output directory (default: ../output)
        enrichment_dir: Directory for earnings data (default: ../enrichment)
        min_nonzero_frac: Minimum fraction of non-zero values required (default: 1.0)
        period: Period for seasonality (default: 53)
        seasonal: Seasonal smoothing parameter (default: 53)
        trend: Trend smoothing parameter (default: 103)
        event_threshold_high: Start positive event when r_cal_t >= this (default: 3.0)
        event_stop_pos: Stop positive event after r_cal_t <= this (default: 1.5)
        start_consec: Require this many consecutive points to start an event (default: 1)
        end_consec: Require this many consecutive points past stop threshold to end an event (default: 2)
        pre: Weeks before start for baseline (default: 8)
        post: Weeks after end for baseline (default: 8)
        tail: Weeks after peak for retention (default: 4)
        shift_thr: r-units threshold for regime_shift_flag_e (default: 0.5)
        require_min_change: If True, only create event when at least one value in the event
            window differs from the pre-event average by min_change_fraction (default: False)
        min_change_fraction: Minimum relative change vs pre-event average when
            require_min_change is True (default: 0.10 = 10%)
        pre_weeks_for_avg: Number of weeks before event start for baseline average (default: 4)
        save_events: Whether to save the event table (default: True)
        save_quarter_events: Whether to save the quarter events table (default: True)
        events_filename: Optional filename for events table (default: None, uses timestamp)
        quarter_events_filename: Filename for quarter events table (default: "quarter_events")
        verbose: Whether to print progress (default: True)
        test_keyword_limit: Limit number of keywords to process (for testing, default: None)
        n_jobs: Number of parallel jobs for keyword processing (default: None = use all CPUs, 1 = sequential)
    
    Returns:
        tuple: (events_df, quarter_events_df)
    """
    pipeline_start = time.time()
    if verbose:
        print("="*70)
        print(f"[PIPELINE] Starting full pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        print(f"[PIPELINE] Parameters:")
        print(f"  event_threshold_high: {event_threshold_high}")
        print(f"  event_stop_pos: {event_stop_pos}")
        print(f"  start_consec: {start_consec}")
        print(f"  end_consec: {end_consec}")
        print(f"  period: {period}, seasonal: {seasonal}, trend: {trend}")
        print(f"  require_min_change: {require_min_change}, min_change_fraction: {min_change_fraction}, pre_weeks_for_avg: {pre_weeks_for_avg}")
        if test_keyword_limit:
            print(f"  ⚠️  TEST MODE: Limited to {test_keyword_limit} keywords")
        if n_jobs:
            print(f"  Parallel workers: {n_jobs}")
        print("="*70)
    
    if output_dir is None:
        output_dir = Path(".") / "data"
    else:
        output_dir = Path(output_dir)
    
    # Load time series data
    if verbose:
        print(f"[PIPELINE] Step 1/3: Loading time series data from {output_dir}")
    ts_files = sorted(output_dir.glob("normalized_timeseries_branded_*_part*.csv"))
    if not ts_files:
        raise FileNotFoundError("No normalized_timeseries_branded_*_part*.csv found in output/")
    
    if verbose:
        print(f"[PIPELINE] Found {len(ts_files)} time series files")
    
    load_start = time.time()
    df_all = pd.concat([pd.read_csv(f) for f in ts_files], ignore_index=True)
    if verbose:
        print(f"[PIPELINE] Loaded {len(df_all)} rows in {time.time() - load_start:.2f} seconds")
        print(f"[PIPELINE] Unique keywords: {df_all['keyword'].nunique()}")
    
    # Build event table
    if verbose:
        print(f"\n[PIPELINE] Step 2/3: Building event table")
    
    # Set test limit if specified
    if test_keyword_limit:
        build_event_table._test_limit = test_keyword_limit
    
    try:
        events_df = build_event_table(
            df_all,
            min_nonzero_frac=min_nonzero_frac,
            period=period,
            seasonal=seasonal,
            trend=trend,
            event_threshold_high=event_threshold_high,
            event_stop_pos=event_stop_pos,
            start_consec=start_consec,
            end_consec=end_consec,
            pre=pre,
            post=post,
            tail=tail,
            shift_thr=shift_thr,
            require_min_change=require_min_change,
            min_change_fraction=min_change_fraction,
            pre_weeks_for_avg=pre_weeks_for_avg,
            output_dir=output_dir,
            save=save_events,
            filename=events_filename,
            verbose=verbose,
            n_jobs=n_jobs
        )
    finally:
        # Clean up test limit
        if hasattr(build_event_table, '_test_limit'):
            delattr(build_event_table, '_test_limit')
    
    if verbose:
        print(f"[PIPELINE] Event table built: {len(events_df)} events")
    
    # Merge with earnings
    if verbose:
        print(f"\n[PIPELINE] Step 3/3: Merging with earnings reports")
    quarter_events_df = merge_earnings_reports(
        events_df,
        output_dir=output_dir,
        enrichment_dir=enrichment_dir,
        save=save_quarter_events,
        filename=quarter_events_filename,
        verbose=verbose
    )
    
    if verbose:
        total_time = time.time() - pipeline_start
        print("="*70)
        print(f"[PIPELINE] Pipeline completed in {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
        print(f"[PIPELINE] Final results: {len(events_df)} events, {len(quarter_events_df)} quarter events")
        print("="*70)
    
    return events_df, quarter_events_df


if __name__ == "__main__":
    events_df, quarter_events_df = run_full_pipeline()
    print(f"Events: {len(events_df)}")
    print(f"Quarter events: {len(quarter_events_df)}")

