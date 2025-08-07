#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tf
from tifffile import imread
from skimage.io import imread
from skimage.measure import label, regionprops_table
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
from scipy.optimize import curve_fit


# In[2]:


DATA_DIR = "/Users/asus/Downloads/MSc Final Project Stuff/2D_processed_time_lapse_datasets"
wt_dir   = os.path.join(DATA_DIR, "WT")
mut_dir  = os.path.join(DATA_DIR, "MUT")

# List all .tif stacks
wt_files  = sorted(glob(os.path.join(wt_dir,  "*.tif")))
mut_files = sorted(glob(os.path.join(mut_dir, "*.tif")))

# Load marker quantifications
csv_path = "/Users/asus/Downloads/MSc Final Project Stuff/CodeBuilding/tert_muSC-quantification_all.csv"
quant_df = pd.read_csv(csv_path)

# Print counts and CSV shape
print(f"Found {len(wt_files)} WT stacks")
print(f"Found {len(mut_files)} MUT stacks")
print(f"Quant CSV: {quant_df.shape[0]} rows × {quant_df.shape[1]} columns\n")


# In[3]:


# Load & Inspect One WT and One MUT Stack

# Pick the first file from each list
wt_sample = wt_files[0]
mut_sample = mut_files[0]

# Read the stacks
wt_stack  = imread(wt_sample)
mut_stack = imread(mut_sample)

# Print basic info
def summarize_stack(name, stack, path):
    print(f"{name} stack: {os.path.basename(path)}")
    print(f"  • Shape (frames, height, width): {stack.shape}")
    print(f"  • Data type: {stack.dtype}")
    print(f"  • Pixel range: {stack.min()} – {stack.max()}\n")

summarize_stack("WT",  wt_stack,  wt_sample)
summarize_stack("MUT", mut_stack, mut_sample)

# Display Frame 0 of each
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(wt_stack[0], cmap='gray')
axes[0].set_title(f"WT (Frame 0)")
axes[1].imshow(mut_stack[0], cmap='gray')
axes[1].set_title(f"MUT (Frame 0)")
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()


# In[4]:


# Object Detection

# Prepare list of all (file_path, genotype) pairs
all_stacks = (
    [(p, "WT") for p in wt_files] +
    [(p, "MUT") for p in mut_files]
)

# Loop over each stack & each frame, extract shape & centroid features in one go
records = []
for stack_path, genotype in all_stacks:
    stack = tf.imread(stack_path)                 # fast multipage TIFF read
    stack_name = os.path.basename(stack_path)
    for frame_idx, frame in enumerate(stack):
        mask = frame > 0                           # binary mask of cells
        labeled = label(mask)
        props = regionprops_table(
            labeled,
            properties=[
                "label", "area", "perimeter",
                "eccentricity", "solidity", "orientation",
                "centroid"
            ]
        )
        df = pd.DataFrame(props)
        # Rename centroid columns for clarity
        df = df.rename(columns={
            "centroid-0": "centroid_y",
            "centroid-1": "centroid_x"
        })
        # Annotate with stack info
        df["stack"]    = stack_name
        df["frame"]    = frame_idx
        df["Genotype"] = genotype
        records.append(df)

# Concatenate all records into one DataFrame
objects_df = pd.concat(records, ignore_index=True)

# Sanity checks
print("objects_df shape:", objects_df.shape)
print("Columns:", objects_df.columns.tolist())
print("\nSample rows:")
print(objects_df.head())


# In[5]:


# Side‑by‑Side Overlay for WT & MUT Frame 0 

# Identify the first sample from each genotype
wt_sample_name = os.path.basename(wt_files[0]) 
mut_sample_name = os.path.basename(mut_files[0])

# Extract detections for frame 0 of each
wt_sample0  = objects_df[(objects_df['stack'] == wt_sample_name)  & (objects_df['frame'] == 0)]
mut_sample0 = objects_df[(objects_df['stack'] == mut_sample_name) & (objects_df['frame'] == 0)]

# Print counts
print(f"{wt_sample_name} — Frame 0 detected objects: {len(wt_sample0)}")
print(f"{mut_sample_name} — Frame 0 detected objects: {len(mut_sample0)}\n")

print("WT sample (first 5 rows):")
print(wt_sample0[['label','centroid_y','centroid_x','area']].head(), "\n")

print("MUT sample (first 5 rows):")
print(mut_sample0[['label','centroid_y','centroid_x','area']].head())

# Load Frame 0 images
wt_img0  = imread(wt_files[0])[0]
mut_img0 = imread(mut_files[0])[0]

# Plot side‑by‑side overlays
fig, axes = plt.subplots(1, 2, figsize=(10,5))

axes[0].imshow(wt_img0, cmap='gray')
axes[0].scatter(
    wt_sample0['centroid_x'], wt_sample0['centroid_y'],
    s=40, edgecolor='yellow', facecolor='none', linewidth=1.2
)
axes[0].set_title(f"{wt_sample_name}\nFrame 0 (WT)")
axes[0].axis('off')

axes[1].imshow(mut_img0, cmap='gray')
axes[1].scatter(
    mut_sample0['centroid_x'], mut_sample0['centroid_y'],
    s=40, edgecolor='yellow', facecolor='none', linewidth=1.2
)
axes[1].set_title(f"{mut_sample_name}\nFrame 0 (MUT)")
axes[1].axis('off')

plt.tight_layout()
plt.show()


# In[6]:


# Link Centroids into Tracks

def link_tracks(objects_df, max_disp=30):
    """
    Link objects frame‐to‐frame by nearest‐neighbor within max_disp pixels.
    Returns a DataFrame with columns:
      - All original object columns
      - track_id: unique integer per trajectory
      - dx, dy: per‐step displacements (pixels)
      - step_distance: Euclidean step length
      - stationary: True if step_distance ≈ 0
    """
    tracks = []
    next_track_id = 0

    # Process each movie (stack) separately
    for stack_name, df_stack in objects_df.groupby('stack', sort=False):
        # sort frames
        frames = sorted(df_stack['frame'].unique())
        prev_coords = None
        prev_ids = None

        for frame in frames:
            df_frame = df_stack[df_stack['frame'] == frame].copy()
            coords = df_frame[['centroid_y', 'centroid_x']].values

            if frame == frames[0]:
                # initialize new tracks for first frame
                ids = np.arange(next_track_id, next_track_id + len(df_frame))
                next_track_id += len(df_frame)
            else:
                # build KD‐tree of previous frame centroids
                tree = cKDTree(prev_coords)
                dists, idxs = tree.query(coords, distance_upper_bound=max_disp)

                ids = []
                for dist, idx in zip(dists, idxs):
                    if idx < len(prev_ids) and dist <= max_disp:
                        ids.append(prev_ids[idx])           # continue existing track
                    else:
                        ids.append(next_track_id)           # start new track
                        next_track_id += 1

            df_frame['track_id'] = ids
            tracks.append(df_frame)

            # update for next iteration
            prev_coords = coords
            prev_ids = ids

    # concatenate all frames & stacks
    tracks_df = pd.concat(tracks, ignore_index=True)

    # compute per‑track step displacements
    tracks_df = tracks_df.sort_values(['stack', 'track_id', 'frame'])
    disp = tracks_df.groupby(['stack','track_id'])[['centroid_x','centroid_y']].diff().fillna(0)
    tracks_df['dx'] = disp['centroid_x']
    tracks_df['dy'] = disp['centroid_y']
    tracks_df['step_distance'] = np.hypot(tracks_df['dx'], tracks_df['dy'])
    tracks_df['stationary'] = tracks_df['step_distance'] < 1e-6

    return tracks_df

# Run linking on your master objects_df
tracks_df = link_tracks(objects_df, max_disp=30)

# Sanity checks
print("tracks_df shape:", tracks_df.shape)
print("Unique tracks:", tracks_df['track_id'].nunique())
print("Average points per track:", 
      tracks_df.groupby('track_id').size().mean().round(1))

# Peek at the first few rows
tracks_df.head()


# In[7]:


# Displacement & Numbered Trajectory for WT and MUT

# Identify sample stack names and pick the longest track in each
wt_name  = os.path.basename(wt_files[0])  
mut_name = os.path.basename(mut_files[0])

wt_tracks  = tracks_df[tracks_df['stack'] == wt_name]
mut_tracks = tracks_df[tracks_df['stack'] == mut_name]

# choose the track_id with the most frames
wt_tid  = wt_tracks['track_id'].value_counts().idxmax()
mut_tid = mut_tracks['track_id'].value_counts().idxmax()

# Extract the full trajectory DataFrames, sorted by frame
wt_traj  = wt_tracks[wt_tracks['track_id'] == wt_tid].sort_values('frame')
mut_traj = mut_tracks[mut_tracks['track_id'] == mut_tid].sort_values('frame')

# Determine start & end points
start_wt, end_wt   = wt_traj.iloc[0], wt_traj.iloc[-1]
start_mut, end_mut = mut_traj.iloc[0], mut_traj.iloc[-1]

# Load the first frame images as background
wt_img0  = imread(wt_files[0])[0]
mut_img0 = imread(mut_files[0])[0]

# ── WT Displacement Image ──
plt.figure(figsize=(5,5))
plt.imshow(wt_img0, cmap='gray')
# start/end markers
plt.scatter(start_wt.centroid_x, start_wt.centroid_y,
            s=100, facecolor='green', edgecolor='black', label='Start')
plt.scatter(end_wt.centroid_x, end_wt.centroid_y,
            s=100, facecolor='red',   edgecolor='black', label='End')
# net displacement arrow
plt.arrow(
    start_wt.centroid_x, start_wt.centroid_y,
    end_wt.centroid_x - start_wt.centroid_x,
    end_wt.centroid_y - start_wt.centroid_y,
    color='cyan', head_width=5, length_includes_head=True, linewidth=2
)
plt.title(f"WT Track {wt_tid}: Net Displacement")
plt.axis('off')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# ── WT Numbered Trajectory Plot ──
plt.figure(figsize=(5,5))
plt.plot(wt_traj.centroid_x, wt_traj.centroid_y, '-o',
         color='cyan', linewidth=2, markersize=6)
for _, row in wt_traj.iterrows():
    plt.text(row.centroid_x + 1, row.centroid_y + 1,
             str(int(row.frame)),
             color='white', fontsize=8,
             ha='center', va='center',
             bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
# start/end markers
plt.scatter(start_wt.centroid_x, start_wt.centroid_y,
            s=100, facecolor='green', edgecolor='black', label='Start')
plt.scatter(end_wt.centroid_x, end_wt.centroid_y,
            s=100, facecolor='red',   edgecolor='black', label='End')
plt.gca().invert_yaxis()
plt.title(f"WT Track {wt_tid}: Numbered Trajectory")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ── MUT Displacement Image ──
plt.figure(figsize=(5,5))
plt.imshow(mut_img0, cmap='gray')
plt.scatter(start_mut.centroid_x, start_mut.centroid_y,
            s=100, facecolor='green', edgecolor='black', label='Start')
plt.scatter(end_mut.centroid_x, end_mut.centroid_y,
            s=100, facecolor='red',   edgecolor='black', label='End')
plt.arrow(
    start_mut.centroid_x, start_mut.centroid_y,
    end_mut.centroid_x - start_mut.centroid_x,
    end_mut.centroid_y - start_mut.centroid_y,
    color='magenta', head_width=5, length_includes_head=True, linewidth=2
)
plt.title(f"MUT Track {mut_tid}: Net Displacement")
plt.axis('off')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# ── MUT Numbered Trajectory Plot ──
plt.figure(figsize=(5,5))
plt.plot(mut_traj.centroid_x, mut_traj.centroid_y, '-o',
         color='magenta', linewidth=2, markersize=6)
for _, row in mut_traj.iterrows():
    plt.text(row.centroid_x + 1, row.centroid_y + 1,
             str(int(row.frame)),
             color='white', fontsize=8,
             ha='center', va='center',
             bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
plt.scatter(start_mut.centroid_x, start_mut.centroid_y,
            s=100, facecolor='green', edgecolor='black', label='Start')
plt.scatter(end_mut.centroid_x, end_mut.centroid_y,
            s=100, facecolor='red',   edgecolor='black', label='End')
plt.gca().invert_yaxis()
plt.title(f"MUT Track {mut_tid}: Numbered Trajectory")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[8]:


# Compute Per‑Track Movement Metrics 

def compute_track_metrics(df):
    """
    Given one track's DataFrame (sorted by frame), compute:
      - num_frames: number of points in the track
      - total_path_length: sum of step distances
      - net_displacement: straight‑line distance from first to last point
      - directionality_ratio: net_displacement / total_path_length
      - mean_speed, max_speed (px per frame)
      - n_stationary: count of frames with zero movement
      - n_lost: placeholder (0) — can be added if linking flags 'lost'
      - persistence_fraction: same as directionality_ratio
    """
    total_path = df['step_distance'].sum()
    net_disp = np.hypot(
        df['centroid_x'].iloc[-1] - df['centroid_x'].iloc[0],
        df['centroid_y'].iloc[-1] - df['centroid_y'].iloc[0]
    )
    directionality = net_disp / total_path if total_path > 0 else np.nan
    
    return pd.Series({
        'num_frames':           df.shape[0],
        'total_path_length':    total_path,
        'net_displacement':     net_disp,
        'directionality_ratio': directionality,
        'mean_speed':           df['step_distance'].mean(),
        'max_speed':            df['step_distance'].max(),
        'n_stationary':         int(df['stationary'].sum()),
        'n_lost':               0,
        'persistence_fraction': directionality
    })

# Apply to each track
track_metrics = (
    tracks_df
    .sort_values(['stack','track_id','frame'])
    .groupby(['stack','track_id','Genotype'], sort=False)
    .apply(compute_track_metrics)
    .reset_index()
)

# Inspect the first few rows
print("Per-track metrics:")
print(track_metrics.head())


# In[9]:


# Boxplots: Mean Speed and Persistence by Genotype
fig, axes = plt.subplots(1, 2, figsize=(10,4))

# Mean Speed
axes[0].boxplot([
    track_metrics[track_metrics.Genotype=='WT']['mean_speed'].dropna(),
    track_metrics[track_metrics.Genotype=='MUT']['mean_speed'].dropna()
], labels=['WT','MUT'])
axes[0].set_title('Mean Speed')
axes[0].set_ylabel('px / frame')

# Persistence Fraction
axes[1].boxplot([
    track_metrics[track_metrics.Genotype=='WT']['persistence_fraction'].dropna(),
    track_metrics[track_metrics.Genotype=='MUT']['persistence_fraction'].dropna()
], labels=['WT','MUT'])
axes[1].set_title('Persistence Fraction')
axes[1].set_ylabel('Net / Total Path')

plt.suptitle('Movement Metrics: WT vs MUT')
plt.tight_layout()
plt.show()


# In[10]:


# Compute Per‑Track Shape Dynamics 

def compute_shape_dynamics(df, feature):
    """
    For one track’s DataFrame (sorted by frame) and one shape feature,
    compute:
      - {feature}_mean: average of the feature over time
      - {feature}_delta_mean: mean of frame‑to‑frame differences (Δfeature)
      - {feature}_delta_max: maximum absolute single‑step change
      - {feature}_slope: slope of the linear fit feature vs frame
    """
    vals   = df[feature].values
    frames = df['frame'].values
    deltas = np.diff(vals)
    
    slope = np.polyfit(frames, vals, 1)[0] if len(vals) > 1 else np.nan
    
    return {
        f'{feature}_mean':        np.mean(vals),
        f'{feature}_delta_mean':  np.mean(deltas) if deltas.size>0 else 0,
        f'{feature}_delta_max':   np.max(np.abs(deltas)) if deltas.size>0 else 0,
        f'{feature}_slope':       slope
    }

# Define all shape features to process
shape_features = ['area', 'perimeter', 'eccentricity', 'solidity', 'orientation']

# Loop over each track and assemble dynamics records
records = []
for (stack, tid, geno), df_tr in tracks_df.groupby(['stack','track_id','Genotype'], sort=False):
    dyn = {'stack': stack, 'track_id': tid, 'Genotype': geno}
    for feat in shape_features:
        dyn.update(compute_shape_dynamics(df_tr.sort_values('frame'), feat))
    records.append(dyn)

shape_dynamics_df = pd.DataFrame(records)

# Inspect the first few rows
print("Shape dynamics (first 6 rows):")
print(shape_dynamics_df.head())


# In[11]:


# Shape Dynamics Visualizations — Perimeter

# Identify sample stacks and the longest track per genotype
wt_name  = os.path.basename(wt_files[0])
mut_name = os.path.basename(mut_files[0])

wt_tracks  = tracks_df[tracks_df['stack'] == wt_name]
mut_tracks = tracks_df[tracks_df['stack'] == mut_name]

wt_tid  = wt_tracks['track_id'].value_counts().idxmax()
mut_tid = mut_tracks['track_id'].value_counts().idxmax()

# Extract perimeter time series for these tracks
wt_traj  = wt_tracks[wt_tracks['track_id'] == wt_tid].sort_values('frame')
mut_traj = mut_tracks[mut_tracks['track_id'] == mut_tid].sort_values('frame')

# Statistical summaries from shape_dynamics_df
wt_stats  = shape_dynamics_df[shape_dynamics_df['Genotype']=='WT']
mut_stats = shape_dynamics_df[shape_dynamics_df['Genotype']=='MUT']

# Create a 2×2 figure
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Top‑left: WT perimeter vs. frame
axes[0,0].plot(wt_traj['frame'], wt_traj['perimeter'], '-o', color='cyan')
axes[0,0].set_title(f"WT Track {wt_tid}: Perimeter over Time")
axes[0,0].set_xlabel("Frame")
axes[0,0].set_ylabel("Perimeter (px)")

# Top‑right: MUT perimeter vs. frame
axes[0,1].plot(mut_traj['frame'], mut_traj['perimeter'], '-o', color='magenta')
axes[0,1].set_title(f"MUT Track {mut_tid}: Perimeter over Time")
axes[0,1].set_xlabel("Frame")
axes[0,1].set_ylabel("Perimeter (px)")

# Bottom‑left: Boxplot of perimeter_mean by genotype
axes[1,0].boxplot(
    [wt_stats['perimeter_mean'].dropna(), mut_stats['perimeter_mean'].dropna()],
    labels=['WT','MUT']
)
axes[1,0].set_title("Distribution of Avg. Perimeter")
axes[1,0].set_ylabel("Perimeter_mean (px)")

# Bottom‑right: Boxplot of perimeter_slope by genotype
axes[1,1].boxplot(
    [wt_stats['perimeter_slope'].dropna(), mut_stats['perimeter_slope'].dropna()],
    labels=['WT','MUT']
)
axes[1,1].set_title("Distribution of Perimeter Slope")
axes[1,1].set_ylabel("Perimeter_slope (px/frame)")

plt.suptitle("Shape Dynamics: Perimeter (WT vs MUT)", y=0.95)
plt.tight_layout()
plt.show()


# In[12]:


# Sample Area & Eccentricity Dynamics + Population Distributions 

# Pick longest track from each genotype
wt_name, mut_name = os.path.basename(wt_files[0]), os.path.basename(mut_files[0])
wt_tracks  = tracks_df[tracks_df['stack'] == wt_name]
mut_tracks = tracks_df[tracks_df['stack'] == mut_name]
wt_tid  = wt_tracks['track_id'].value_counts().idxmax()
mut_tid = mut_tracks['track_id'].value_counts().idxmax()

# Extract trajectories (sorted by frame)
wt_traj  = wt_tracks[wt_tracks['track_id']==wt_tid].sort_values('frame')
mut_traj = mut_tracks[mut_tracks['track_id']==mut_tid].sort_values('frame')

# Gather population statistics
wt_stats  = shape_dynamics_df[shape_dynamics_df['Genotype']=='WT']
mut_stats = shape_dynamics_df[shape_dynamics_df['Genotype']=='MUT']

# ── A) area & eccentricity
fig, axes = plt.subplots(2, 2, figsize=(10,8))

# WT area vs frame
axes[0,0].plot(wt_traj['frame'], wt_traj['area'], '-o', color='cyan')
axes[0,0].set_title(f"WT Track {wt_tid}: Area over Time")
axes[0,0].set_xlabel("Frame")
axes[0,0].set_ylabel("Area (px²)")

# WT eccentricity vs frame
axes[1,0].plot(wt_traj['frame'], wt_traj['eccentricity'], '-o', color='cyan')
axes[1,0].set_title(f"WT Track {wt_tid}: Eccentricity over Time")
axes[1,0].set_xlabel("Frame")
axes[1,0].set_ylabel("Eccentricity")

# MUT area vs frame
axes[0,1].plot(mut_traj['frame'], mut_traj['area'], '-o', color='magenta')
axes[0,1].set_title(f"MUT Track {mut_tid}: Area over Time")
axes[0,1].set_xlabel("Frame")
axes[0,1].set_ylabel("Area (px²)")

# MUT eccentricity vs frame
axes[1,1].plot(mut_traj['frame'], mut_traj['eccentricity'], '-o', color='magenta')
axes[1,1].set_title(f"MUT Track {mut_tid}: Eccentricity over Time")
axes[1,1].set_xlabel("Frame")
axes[1,1].set_ylabel("Eccentricity")

plt.tight_layout()
plt.show()

# ── B) Population distributions: mean & slope
fig, axes = plt.subplots(2, 2, figsize=(10,8))

# Avg. Area
axes[0,0].boxplot([wt_stats['area_mean'], mut_stats['area_mean']], labels=['WT','MUT'])
axes[0,0].set_title("Average Area per Track")
axes[0,0].set_ylabel("Area_mean (px²)")

# Area slope
axes[0,1].boxplot([wt_stats['area_slope'], mut_stats['area_slope']], labels=['WT','MUT'])
axes[0,1].set_title("Area Trend per Track")
axes[0,1].set_ylabel("Area_slope (px²/frame)")

# Avg. Eccentricity
axes[1,0].boxplot([wt_stats['eccentricity_mean'], mut_stats['eccentricity_mean']], labels=['WT','MUT'])
axes[1,0].set_title("Average Eccentricity per Track")
axes[1,0].set_ylabel("Eccentricity_mean")

# Eccentricity slope
axes[1,1].boxplot([wt_stats['eccentricity_slope'], mut_stats['eccentricity_slope']], labels=['WT','MUT'])
axes[1,1].set_title("Eccentricity Trend per Track")
axes[1,1].set_ylabel("Eccentricity_slope")

plt.tight_layout()
plt.show()


# In[13]:


# Aggregate to Per‑File Summaries (wt_all & mut_all)

# Combine movement metrics and shape dynamics per track
tracks_full = (
    track_metrics
    .merge(
        shape_dynamics_df,
        on=['stack','track_id','Genotype'],
        how='left'
    )
    .rename(columns={'stack':'source_file'})
)

# Define which columns to aggregate
agg_columns = [
    'mean_speed', 'max_speed', 'persistence_fraction',
    'area_mean', 'eccentricity_mean',
    'perimeter_mean', 'solidity_mean'
]

# Aggregate per source_file & Genotype: compute mean and std
per_file = (
    tracks_full
    .groupby(['source_file','Genotype'])[agg_columns]
    .agg(['mean','std'])
)

# Flatten the MultiIndex columns
per_file.columns = [
    f"{col}_{stat}" for col, stat in per_file.columns
]
per_file = per_file.reset_index()

# Split into WT and MUT tables
wt_all  = per_file[per_file.Genotype == 'WT'].copy()
mut_all = per_file[per_file.Genotype == 'MUT'].copy()

# Sanity check
print("wt_all shape:", wt_all.shape)
print("mut_all shape:", mut_all.shape)
display(wt_all.head(), mut_all.head())


# In[14]:


# Identify all per‑track “mean” columns in shape_dynamics_df
shape_means = [
    col for col in shape_dynamics_df.columns 
    if col.endswith('_mean')
    and col not in ['area_mean','eccentricity_mean']
]

# Build the per-track table
per_track = (
    track_metrics
    .merge(
        shape_dynamics_df[['stack','track_id','Genotype'] + shape_means + 
                          ['area_mean','eccentricity_mean']],
        on=['stack','track_id','Genotype'], how='left'
    )
    .assign(directionality_ratio = lambda df: df['persistence_fraction'])
)

# Define the full list of metrics to test
movement_metrics = [
    'mean_speed',
    'net_displacement',
    'directionality_ratio',
    'persistence_fraction'
]
shape_metrics = ['area_mean','eccentricity_mean'] + shape_means
all_metrics = movement_metrics + shape_metrics

# Filter valid tracks (≥2 frames & movement >0)
valid = (track_metrics['num_frames'] > 1) & (track_metrics['total_path_length'] > 0)
pt = per_track[valid]

# Split WT vs MUT
wt = pt[pt.Genotype=='WT']
mut= pt[pt.Genotype=='MUT']

# Run Bonferroni‐corrected Mann–Whitney U over all metrics
alpha_corr = 0.05 / len(all_metrics)
results = []
for m in all_metrics:
    u, p = mannwhitneyu(wt[m], mut[m], alternative='two-sided')
    results.append({
        'Metric':      m,
        'WT Median':   np.median(wt[m]),
        'MUT Median':  np.median(mut[m]),
        'U-value':     u,
        'p-value':     p,
        'Significant': 'Yes' if p < alpha_corr else 'No'
    })

results_df = pd.DataFrame(results)
print(f"Tests on {len(all_metrics)} metrics (Bonferroni α={alpha_corr:.3f}):")
display(results_df)


# In[15]:


# Build a lookup of p‑values by metric
p_vals = dict(zip(results_df['Metric'], results_df['p-value']))

# Select the significant metrics and their labels
sig_metrics = ['mean_speed', 'eccentricity_mean', 'orientation_mean']
labels      = ['Mean Speed (px/frame)', 'Mean Eccentricity', 'Mean Orientation (rad)']

# Extract per‑track data for WT/MUT
wt = per_track[per_track.Genotype == 'WT']
mut= per_track[per_track.Genotype == 'MUT']

# Compute Bonferroni threshold
alpha_corr = 0.05 / len(results_df)

# Plot side‑by‑side boxplots with jitter + p‑value annotation
fig, axes = plt.subplots(1, len(sig_metrics), figsize=(15,5))
fig.subplots_adjust(top=0.85, wspace=0.4)

for ax, metric, label in zip(axes, sig_metrics, labels):
    # boxplot
    ax.boxplot([wt[metric], mut[metric]], labels=['WT','MUT'])
    # jittered points
    x_wt = np.random.normal(1, 0.05, size=len(wt))
    x_mut= np.random.normal(2, 0.05, size=len(mut))
    ax.scatter(x_wt, wt[metric], color='cyan', edgecolor='k', alpha=0.7)
    ax.scatter(x_mut, mut[metric], color='magenta', edgecolor='k', alpha=0.7)
    
    # annotate p‑value
    p = p_vals[metric]
    star = ' *' if p < alpha_corr else ''
    ymax = max(wt[metric].max(), mut[metric].max())
    ax.text(
        1.5, ymax * 1.05,
        f"p = {p:.1e}{star}",
        ha='center', va='bottom',
        fontsize=12
    )
    
    # labels
    ax.set_title(label)
    ax.set_ylabel(label)
    ax.set_xlabel('Genotype')

plt.suptitle("Significant Per‑Track Differences: WT vs MUT", fontsize=16)
plt.show()


# In[16]:


# ─── Step 3: Per‑Track Diffusion Analysis (MSD + Model Fits) ───

import numpy as np
from scipy.optimize import curve_fit

# Build per‑track trajectories from tracks_df
tracks = {
    tid: grp.sort_values('frame')[['centroid_x','centroid_y']].values
    for tid, grp in tracks_df.groupby('track_id')
}

# Short‑lag MSD function
def msd(traj, max_lag=10):
    n = len(traj)
    lags = min(max_lag, n-1)
    vals = []
    for lag in range(1, lags+1):
        disp = traj[lag:] - traj[:-lag]
        vals.append(np.sum(disp**2, axis=1).mean())
    return np.array(vals)

# Power‑law fit: log(MSD)=logK + α log(τ)
def fit_power_law(msd_vals):
    if len(msd_vals) < 2:
        return np.nan, np.nan
    taus = np.arange(1, len(msd_vals)+1)
    α, logK = np.polyfit(np.log(taus), np.log(msd_vals), 1)
    return α, np.exp(logK)

plaw_results = []
for tid, traj in tracks.items():
    m = msd(traj, max_lag=10)
    α, K = fit_power_law(m)
    plaw_results.append({'track_id': tid, 'alpha': α, 'K': K})
plaw_df = pd.DataFrame(plaw_results)

# PRW fit: MSD(t)=4D[t−P(1−e^{−t/P})]
def msd_prw(t, D, P):
    return 4*D*(t - P*(1 - np.exp(-t/P)))

prw_results = []
for tid, traj in tracks.items():
    m = msd(traj, max_lag=10)
    taus = np.arange(1, len(m)+1)
    if len(m) < 2:
        prw_results.append({'track_id': tid, 'D_est': np.nan, 'P_est': np.nan})
        continue
    D0 = m.mean()/(4*taus.mean()) if taus.mean()>0 else np.nan
    P0 = 1.0
    try:
        popt, _ = curve_fit(msd_prw, taus, m, p0=[D0,P0], maxfev=5000)
        D_fit, P_fit = popt
    except:
        D_fit, P_fit = np.nan, np.nan
    prw_results.append({'track_id': tid, 'D_est': D_fit, 'P_est': P_fit})
prw_df = pd.DataFrame(prw_results)

# Merge diffusion parameters into track_metrics
track_metrics = (
    track_metrics
    .merge(plaw_df, on='track_id', how='left')
    .merge(prw_df,  on='track_id', how='left')
)

# Summarize the new columns
print("Diffusion parameter summary:")
display(track_metrics[['alpha','K','D_est','P_est']].describe())


# In[17]:


# ─── Step 4: Compare Diffusion Parameters Between WT and MUT ───

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# Extract the diffusion parameters per genotype
wt = track_metrics[track_metrics.Genotype == 'WT']
mut= track_metrics[track_metrics.Genotype == 'MUT']

metrics = ['alpha', 'D_est', 'P_est']
labels  = ['Anomalous Exponent (α)', 'Diffusion Coefficient (D_est)', 'Persistence Time (P_est)']

# Mann–Whitney U tests with Bonferroni correction
alpha_level = 0.05
bonf = alpha_level / len(metrics)
results = []

for m in metrics:
    u, p = mannwhitneyu(wt[m].dropna(), mut[m].dropna(), alternative='two-sided')
    results.append({
        'Metric':       m,
        'WT Median':    np.median(wt[m].dropna()),
        'MUT Median':   np.median(mut[m].dropna()),
        'U-value':      u,
        'p-value':      p,
        'Significant':  'Yes' if p < bonf else 'No'
    })

res_df = pd.DataFrame(results)

# Display the test results table
print(f"Mann–Whitney U on diffusion params (Bonferroni α={bonf:.3f}):")
display(res_df)

# Visualize with boxplots + jittered points
fig, axes = plt.subplots(1, 3, figsize=(15,5))
fig.subplots_adjust(wspace=0.4)

for ax, m, lab in zip(axes, metrics, labels):
    # boxplot
    ax.boxplot([wt[m].dropna(), mut[m].dropna()], labels=['WT','MUT'])
    # jittered points
    x_wt = np.random.normal(1, 0.05, size=len(wt))
    x_mut= np.random.normal(2, 0.05, size=len(mut))
    ax.scatter(x_wt, wt[m], color='cyan', edgecolor='k', alpha=0.6)
    ax.scatter(x_mut, mut[m], color='magenta', edgecolor='k', alpha=0.6)
    
    # annotate p‑value
    row = res_df[res_df.Metric == m].iloc[0]
    p   = row['p-value']
    star= ' *' if row['Significant']=='Yes' else ''
    ymax= max(wt[m].max(), mut[m].max())
    ax.text(1.5, ymax * 1.05, f"p={p:.1e}{star}", ha='center')
    
    ax.set_title(lab)
    ax.set_ylabel(lab)
    ax.set_xlabel('Genotype')

plt.suptitle("Per‑Track Diffusion Parameter Comparison: WT vs MUT", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Rebuild per_track with both movement, shape, and diffusion columns
per_track = (
    track_metrics
    .merge(
        shape_dynamics_df[['stack','track_id','Genotype','area_mean','eccentricity_mean']],
        on=['stack','track_id','Genotype'], how='left'
    )
    .merge(plaw_df, on='track_id', how='left')   # alpha, K
    .merge(prw_df,  on='track_id', how='left')   # D_est, P_est
)

# Inspect the column names
print("Columns available for PCA:\n", per_track.columns.tolist())

# Pick your features (only those present)
feature_cols = [
    'mean_speed',
    'persistence_fraction',
    'area_mean',
    'eccentricity_mean',
    'alpha',
    'D_est'
]
# Filter out any that aren’t actually there
feature_cols = [c for c in feature_cols if c in per_track.columns]
print("\nUsing these features:\n", feature_cols)

# Drop rows with NaNs in these features
pca_df = per_track[['Genotype'] + feature_cols].dropna()

# Standardize and run PCA
X = StandardScaler().fit_transform(pca_df[feature_cols])
pca = PCA(n_components=2)
pcs = pca.fit_transform(X)
pc_df = pd.DataFrame(pcs, columns=['PC1','PC2'])
pc_df['Genotype'] = pca_df['Genotype'].values

# Show variance explained
print(f"\nExplained variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
      f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

# Plot PC1 vs PC2
fig, ax = plt.subplots(figsize=(6,6))
for geno, color in [('WT','cyan'), ('MUT','magenta')]:
    sel = pc_df[pc_df.Genotype==geno]
    ax.scatter(sel.PC1, sel.PC2, s=20, alpha=0.6, label=geno, color=color, edgecolor='k')

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.set_title("PCA of Per‑Track Features")
ax.legend(title="Genotype")
plt.tight_layout()
plt.show()


# In[20]:


# ─── PCA with Diffusion Parameters Included ───

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Rebuild per_track with movement, shape, and diffusion
per_track = (
    track_metrics
    .merge(
        shape_dynamics_df[['stack','track_id','Genotype','area_mean','eccentricity_mean']],
        on=['stack','track_id','Genotype'], how='left'
    )
    .merge(plaw_df, on='track_id', how='left')   # adds alpha, K
    .merge(prw_df,  on='track_id', how='left')   # adds D_est, P_est
)

# Choose an expanded feature set
feature_cols = [
    'mean_speed',
    'persistence_fraction',
    'area_mean',
    'eccentricity_mean',
    'alpha',
    'D_est',
    'P_est'
]
# Keep only those columns present
feature_cols = [c for c in feature_cols if c in per_track.columns]

# Drop NaNs and standardize
pca_df = per_track[['Genotype'] + feature_cols].dropna()
X = StandardScaler().fit_transform(pca_df[feature_cols])

# Run PCA (2 components)
pca = PCA(n_components=2)
pcs = pca.fit_transform(X)
pc_df = pd.DataFrame(pcs, columns=['PC1','PC2'])
pc_df['Genotype'] = pca_df['Genotype'].values

# Variance explained
print(f"Explained variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
      f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

# Plot
fig, ax = plt.subplots(figsize=(6,6))
for geno, color in [('WT','cyan'), ('MUT','magenta')]:
    sel = pc_df[pc_df.Genotype==geno]
    ax.scatter(sel.PC1, sel.PC2, s=20, alpha=0.6, label=geno, color=color, edgecolor='k')
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.set_title("PCA with Diffusion + Shape/Motility Features")
ax.legend(title="Genotype")
plt.tight_layout()
plt.show()


# In[23]:


import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

# ───Compute PF‑SDF per File ───

def compute_pair_correlation(centroids, img_shape, max_r=200, bin_width=2):
    dmat      = distance_matrix(centroids, centroids)
    dists     = dmat[np.triu_indices_from(dmat, k=1)]
    bins      = np.arange(0, max_r+bin_width, bin_width)
    counts, edges = np.histogram(dists, bins=bins)
    r         = (edges[:-1] + edges[1:]) / 2
    N         = len(centroids)
    area      = img_shape[0] * img_shape[1]
    rho0      = N / area
    shell_areas = np.pi * ((r + bin_width/2)**2 - (r - bin_width/2)**2)
    g         = (counts / shell_areas) / rho0
    return r, g

pc_rows = []
for fname, grp in tracks_df.groupby('stack'):
    # Use the correct column names
    coords = grp[['centroid_y','centroid_x']].values
    # Pick the appropriate image to get its shape
    img = wt_stack if fname.startswith('WT') else mut_stack
    img_shape = (img.shape[1], img.shape[2])  # (height, width)
    
    # Compute g(r)
    r_vals, g_vals = compute_pair_correlation(coords, img_shape)
    
    # Extract summary statistics
    peak_idx    = np.argmax(g_vals[1:]) + 1
    r_peak      = r_vals[peak_idx]
    g_peak      = g_vals[peak_idx]
    decay_thr   = g_peak / np.e
    below       = np.where(g_vals < decay_thr)[0]
    corr_length = r_vals[below[0]] if len(below)>0 else np.nan
    
    pc_rows.append({
        'source_file': fname,
        'r_vals':      r_vals,
        'g_vals':      g_vals,
        'r_peak':      r_peak,
        'g_peak':      g_peak,
        'corr_length': corr_length
    })

pc_df = pd.DataFrame(pc_rows)

# Preview
print("PF‑SDF summaries:")
display(pc_df[['source_file','r_peak','g_peak','corr_length']])


# In[24]:


import numpy as np
import matplotlib.pyplot as plt

# Stack all r and g arrays
all_r = np.stack(pc_df['r_vals'].values)
all_g = np.stack(pc_df['g_vals'].values)

# Build masks
wt_mask  = pc_df['source_file'].str.startswith('WT')
mut_mask = pc_df['source_file'].str.startswith('MUT')

# Compute mean ± SEM
r       = all_r[0]
mean_wt = np.nanmean(all_g[wt_mask], axis=0)
sem_wt  = np.nanstd(all_g[wt_mask], axis=0) / np.sqrt(wt_mask.sum())
mean_mut= np.nanmean(all_g[mut_mask], axis=0)
sem_mut = np.nanstd(all_g[mut_mask], axis=0) / np.sqrt(mut_mask.sum())

# Plot
plt.figure(figsize=(6,4))
plt.plot(r, mean_wt,  label='WT',  color='cyan')
plt.fill_between(r, mean_wt-sem_wt, mean_wt+sem_wt, color='cyan', alpha=0.3)
plt.plot(r, mean_mut, label='MUT', color='magenta')
plt.fill_between(r, mean_mut-sem_mut, mean_mut+sem_mut, color='magenta', alpha=0.3)

plt.xlabel('Distance r (px)')
plt.ylabel('g(r)')
plt.title('Mean ± SEM Pair‑Correlation g(r): WT vs MUT')
plt.legend()
plt.tight_layout()
plt.show()


# In[25]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# Define metrics and masks
metrics = ['r_peak','g_peak','corr_length']
wt_vals  = {m: pc_df.loc[pc_df.source_file.str.startswith('WT'), m] for m in metrics}
mut_vals = {m: pc_df.loc[pc_df.source_file.str.startswith('MUT'), m] for m in metrics}

alpha_corr = 0.05 / len(metrics)
results_pf = []

# Run tests
for m in metrics:
    u, p = mannwhitneyu(wt_vals[m], mut_vals[m], alternative='two-sided')
    results_pf.append({
        'Metric':       m,
        'WT Median':    np.median(wt_vals[m]),
        'MUT Median':   np.median(mut_vals[m]),
        'U-value':      u,
        'p-value':      p,
        'Significant':  'Yes' if p < alpha_corr else 'No'
    })

res_pf_df = pd.DataFrame(results_pf)

# Display results
print(f"Mann–Whitney U on PF‑SDF summaries (Bonferroni α={alpha_corr:.3f}):")
display(res_pf_df)

# Boxplots
fig, axes = plt.subplots(1, 3, figsize=(12,4))
for ax, m in zip(axes, metrics):
    ax.boxplot([wt_vals[m], mut_vals[m]], labels=['WT','MUT'])
    ax.set_title(m)
    # annotate p‑value
    p = res_pf_df.loc[res_pf_df.Metric==m, 'p-value'].values[0]
    star = ' *' if p < alpha_corr else ''
    ymax = max(wt_vals[m].max(), mut_vals[m].max())
    ax.text(1.5, ymax*1.05, f"p={p:.2e}{star}", ha='center')
plt.suptitle("PF‑SDF Summary Comparison")
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()


# In[26]:


import numpy as np
import pandas as pd

# Compute instantaneous speed per link
tracks_df['speed'] = np.sqrt(tracks_df.dx**2 + tracks_df.dy**2)

# Assign state: stationary (S) if speed ≤1, moving (M) if >1
tracks_df['state'] = np.where(tracks_df['speed'] > 1.0, 'M', 'S')

# Collect transitions between consecutive frames for each track
transitions = []
for tid, grp in tracks_df.sort_values(['stack','track_id','frame']).groupby('track_id'):
    geno = grp['Genotype'].iloc[0]
    states = grp['state'].values
    for prev, nxt in zip(states, states[1:]):
        transitions.append({'Genotype': geno, 'from': prev, 'to': nxt})

trans_df = pd.DataFrame(transitions)

# Compute transition counts and convert to probabilities
count_table = (
    trans_df
    .groupby(['Genotype','from','to'])
    .size()
    .unstack(fill_value=0)
)
prob_table = count_table.div(count_table.sum(axis=1), axis=0)

print("Transition probabilities (rows = state at t, cols = state at t+1):")
display(prob_table)


# In[28]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Prepare survival table
surv = track_metrics[['track_id','num_frames','n_lost','Genotype']].copy()
surv['event'] = (surv['n_lost'] > 0).astype(int)  # 1 if the track “died” mid‐movie

def km_curve(df):
    """
    Compute KM survival curve manually:
    Returns times t and survival probabilities S(t).
    """
    # Sort unique event times
    times = np.sort(df['num_frames'].unique())
    n_tot = len(df)
    S = []
    cum_prod = 1.0
    
    for t in times:
        # number at risk just before time t
        at_risk = (df['num_frames'] >= t).sum()
        # number of events at exactly time t
        d_i = df.loc[df['num_frames']==t, 'event'].sum()
        # update survival
        if at_risk > 0:
            cum_prod *= (1 - d_i / at_risk)
        S.append(cum_prod)
    return times, np.array(S)

# Compute curves for WT and MUT
fig, ax = plt.subplots(figsize=(6,4))
for geno, color in [('WT','cyan'), ('MUT','magenta')]:
    dfg = surv[surv.Genotype == geno]
    t, S = km_curve(dfg)
    ax.step(t, S, where='post', label=geno, color=color)

ax.set_xlabel('Track Duration (frames)')
ax.set_ylabel('Survival Probability')
ax.set_title('Kaplan–Meier Survival: Track Persistence')
ax.legend(title='Genotype')
plt.tight_layout()
plt.show()


# In[30]:


# ─── Rebuild Full Feature Table for Classification ───

# Start with per_track (motility + shape + genotype)
features_df = per_track.copy()

# Merge in diffusion parameters
features_df = (
    features_df
    .merge(plaw_df, on='track_id', how='left')   # adds alpha, K
    .merge(prw_df,  on='track_id', how='left')   # adds D_est, P_est
)

# Merge in PF‑SDF summaries
features_df = features_df.merge(
    pc_df[['source_file','r_peak','g_peak','corr_length']],
    left_on='stack', right_on='source_file', how='left'
)

# Confirm columns
print("Columns available for modeling:")
print(features_df.columns.tolist())

# select the features and genotype label, dropping NaNs
features = [
    'mean_speed','persistence_fraction',
    'area_mean','eccentricity_mean',
    'alpha','D_est','P_est',
    'r_peak','g_peak','corr_length'
]
model_df = features_df.dropna(subset=features + ['Genotype']).copy()

# Extract X and y
X = model_df[features].values
y = (model_df.Genotype == 'MUT').astype(int).values  # WT=0, MUT=1

print(f"Final dataset: {X.shape[0]} tracks × {X.shape[1]} features")


# In[31]:


import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd

# Build pipeline: standardize → L1‑penalized logistic regression with 5‑fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = make_pipeline(
    StandardScaler(),
    LogisticRegressionCV(
        Cs=10,
        cv=cv,
        penalty='l1',
        solver='saga',
        scoring='accuracy',
        max_iter=2000,
        refit=True,
        random_state=42
    )
)

# Fit on the full dataset
model.fit(X, y)

# Training accuracy
train_acc = model.score(X, y)
print(f"Training accuracy: {train_acc:.3f}")

# Cross‑validated accuracy
cv_scores = []
for train_idx, test_idx in cv.split(X, y):
    model.fit(X[train_idx], y[train_idx])
    cv_scores.append(model.score(X[test_idx], y[test_idx]))
print(f"5‑fold CV accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

# Extract and display coefficients
coef = model.named_steps['logisticregressioncv'].coef_.ravel()
feature_names = model_df.columns.intersection(features).tolist()
coef_df = pd.Series(coef, index=feature_names).sort_values(key=abs, ascending=False)
print("\nFeature coefficients (absolute value sorted):")
display(pd.DataFrame({
    'feature': coef_df.index,
    'coefficient': coef_df.values
}))

# Plot coefficients
plt.figure(figsize=(6,4))
coef_df.plot.barh(color='teal')
plt.xlabel('Logistic Regression Coefficient')
plt.title('Feature Importance for Genotype Classification')
plt.tight_layout()
plt.show()


# In[32]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(f"Random Forest 5‑fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("RF feature importances:")
display(importances)


# In[33]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Common setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_all, y_all = X, y  # from previous cell
feature_names = features  # same list

results = {}

# ─── 1. Elastic‑Net Logistic Regression ────────────────────────────────
# Using saga solver with elasticnet penalty
l1_ratios = [0.1, 0.5, 0.9]
elastic = make_pipeline(
    StandardScaler(),
    LogisticRegressionCV(
        Cs=5,
        cv=cv,
        penalty='elasticnet',
        solver='saga',
        l1_ratios=l1_ratios,
        scoring='accuracy',
        max_iter=2000,
        random_state=42,
        refit=True
    )
)

scores_elastic = cross_val_score(elastic, X_all, y_all, cv=cv, scoring='accuracy')
elastic.fit(X_all, y_all)
# Extract the best l1_ratio and coefs
best_l1 = elastic.named_steps['logisticregressioncv'].l1_ratio_[0]
coef_elastic = elastic.named_steps['logisticregressioncv'].coef_.ravel()

results['ElasticNet'] = {
    'cv_mean': np.mean(scores_elastic),
    'cv_std':  np.std(scores_elastic),
    'l1_ratio': best_l1,
    'coefs': coef_elastic
}

# ─── 2. Gradient‑Boosted Trees ────────────────────────────────────────
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

scores_gb = cross_val_score(gb, X_all, y_all, cv=cv, scoring='accuracy')
gb.fit(X_all, y_all)
feat_imp_gb = gb.feature_importances_

results['GBM'] = {
    'cv_mean': np.mean(scores_gb),
    'cv_std':  np.std(scores_gb),
    'feat_imp': feat_imp_gb
}

# ─── 3. Report accuracies ─────────────────────────────────────────────
print(f"Elastic‑Net CV accuracy: {results['ElasticNet']['cv_mean']:.3f} ± {results['ElasticNet']['cv_std']:.3f}")
print(f"GBM CV accuracy:         {results['GBM']['cv_mean']:.3f} ± {results['GBM']['cv_std']:.3f}\n")

# ─── 4. Coefs & Importances ───────────────────────────────────────────
# Elastic‑Net coefficients
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coef_elastic': results['ElasticNet']['coefs']
}).set_index('feature').sort_values('coef_elastic', key=abs, ascending=False)

# GBM importances
imp_df = pd.DataFrame({
    'feature': feature_names,
    'imp_gbm': results['GBM']['feat_imp']
}).set_index('feature').sort_values('imp_gbm', ascending=False)

# Combine side by side
comb_df = pd.concat([coef_df, imp_df], axis=1).fillna(0)
display(comb_df)

# ─── 5. Plot comparison ───────────────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(10,5))
comb_df['coef_elastic'].plot.barh(ax=axes[0], color='teal')
axes[0].set_title(f"Elastic‑Net Coefs (l1_ratio={results['ElasticNet']['l1_ratio']:.2f})")
axes[0].axvline(0, color='black', linewidth=0.8)

comb_df['imp_gbm'].plot.barh(ax=axes[1], color='orange')
axes[1].set_title("GBM Feature Importances")

for ax in axes:
    ax.set_xlabel('')
plt.tight_layout()
plt.show()


# In[34]:


from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=4)
scores_gb = cross_val_score(gb, X_all, y_all, groups=model_df['stack'], cv=gkf, scoring='accuracy')
print(f"GBM (GroupKFold by stack) CV accuracy: {scores_gb.mean():.3f} ± {scores_gb.std():.3f}")


# In[ ]:




