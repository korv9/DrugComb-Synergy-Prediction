#%%
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import gc  # Garbage collection
import glob

def merge_fingerprints_with_synergy(batch_size=250, max_pairs=None):
    """
    Memory-optimized version that merges fingerprints with synergy data in batches
    
    Args:
        batch_size: Number of drug pairs to process in each batch
        max_pairs: Maximum number of pairs to process (None for all)
    """
    # Define paths
    fingerprints_path = "C:/Users/46762/VSCODE/BIG_PHARMA/data/interim/fingerprints/morgan_fingerprints.csv"
    synergy_path = "C:/Users/46762/VSCODE/BIG_PHARMA/data/interim/scored_clean.csv"
    output_dir = "C:/Users/46762/VSCODE/BIG_PHARMA/data/interim/merged_data"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data - using optimized approach
    print("Loading fingerprints data...")
    # Read only drug name and cid columns first
    fingerprints_df_info = pd.read_csv(fingerprints_path, usecols=['drug', 'cid'])
    fingerprints_df_info = fingerprints_df_info.rename(columns={'drug': 'drug_name'})
    
    print("Loading synergy data...")
    synergy_df = pd.read_csv(synergy_path)
    
    print(f"Found {len(fingerprints_df_info)} drugs with fingerprints")
    print(f"Found {len(synergy_df)} drug combinations with synergy scores")
    
    # Create drug name list for fingerprints lookup
    drug_names = set(fingerprints_df_info['drug_name'])
    
    # Filter synergy data to include only drug pairs where both have fingerprints
    print("Pre-filtering synergy data...")
    valid_synergy_mask = synergy_df['drug_min'].isin(drug_names) & synergy_df['drug_max'].isin(drug_names)
    valid_synergy_df = synergy_df[valid_synergy_mask].reset_index(drop=True)
    skipped_synergy_df = synergy_df[~valid_synergy_mask].reset_index(drop=True)
    
    print(f"Valid drug combinations: {len(valid_synergy_df)}")
    print(f"Skipped combinations: {len(skipped_synergy_df)}")
    
    # Optionally limit the number of pairs to process
    if max_pairs is not None and max_pairs < len(valid_synergy_df):
        print(f"Limiting to {max_pairs} drug pairs")
        valid_synergy_df = valid_synergy_df.head(max_pairs).reset_index(drop=True)
    
    # Create a list to store which drugs we actually need to load fingerprints for
    needed_drugs = set(valid_synergy_df['drug_min'].unique()) | set(valid_synergy_df['drug_max'].unique())
    print(f"Need to load fingerprints for {len(needed_drugs)} unique drugs")
    
    # Now load fingerprints only for drugs we need
    print("Loading fingerprints for required drugs...")
    # Get column names first
    col_names = pd.read_csv(fingerprints_path, nrows=0).columns.tolist()
    bit_columns = [col for col in col_names if col.startswith('bit_')]
    
    # Create a dictionary to store fingerprints
    drug_to_fingerprint = {}
    
    # Read the CSV in chunks to reduce memory usage
    chunk_size = 250  # Reduced for less memory usage
    for chunk in tqdm(pd.read_csv(fingerprints_path, chunksize=chunk_size), 
                     desc="Loading fingerprint chunks"):
        chunk = chunk.rename(columns={'drug': 'drug_name'})
        # Filter to include only drugs we need
        chunk = chunk[chunk['drug_name'].isin(needed_drugs)]
        
        # Store fingerprints for these drugs - ensure they're float32 for memory efficiency
        for _, row in chunk.iterrows():
            drug_name = row['drug_name']
            fingerprint = row[bit_columns].values.astype(np.float32)  # Force float32 type
            drug_to_fingerprint[drug_name] = fingerprint
            
            # Remove this drug from the needed set to avoid duplicates
            if drug_name in needed_drugs:
                needed_drugs.remove(drug_name)
        
        # Clear memory
        del chunk
        gc.collect()
    
    print(f"Loaded fingerprints for {len(drug_to_fingerprint)} drugs")
    if len(needed_drugs) > 0:
        print(f"Warning: Could not find fingerprints for {len(needed_drugs)} drugs")
    
    # Free memory
    del fingerprints_df_info
    gc.collect()
    
    # Initialize output directories
    np_dir = os.path.join(output_dir, "numpy_arrays")
    batch_dir = os.path.join(np_dir, "batches")
    os.makedirs(np_dir, exist_ok=True)
    os.makedirs(batch_dir, exist_ok=True)
    
    # Create metadata file
    metadata_file = os.path.join(np_dir, "metadata.csv")
    
    # If metadata file exists, remove it to start fresh
    if os.path.exists(metadata_file):
        os.remove(metadata_file)
        
    # Process synergy data in batches
    print("Processing synergy data in batches...")
    num_batches = (len(valid_synergy_df) + batch_size - 1) // batch_size
    
    # Initialize arrays for compact data
    all_rows = []
    total_processed = 0
    
    # Clear any existing batch files
    for f in glob.glob(os.path.join(batch_dir, "*.npy")):
        os.remove(f)
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(valid_synergy_df))
        
        batch_df = valid_synergy_df.iloc[start_idx:end_idx].copy()
        
        # Temporary arrays for this batch
        batch_X_fp_min = []
        batch_X_fp_max = []
        batch_y_synergy = []
        batch_metadata = []
        
        # For each drug pair, get fingerprints and synergy scores
        for _, row in batch_df.iterrows():
            drug_min = row['drug_min']
            drug_max = row['drug_max']
            
            # Both drugs should have fingerprints due to pre-filtering
            fp_min = drug_to_fingerprint.get(drug_min)
            fp_max = drug_to_fingerprint.get(drug_max)
            
            if fp_min is not None and fp_max is not None:
                # Add to numpy arrays
                batch_X_fp_min.append(fp_min)
                batch_X_fp_max.append(fp_max)
                batch_y_synergy.append(float(row['synergy_zip']))  # Force float type
                
                # Add metadata
                metadata_dict = {
                    'drug_min': drug_min,
                    'drug_max': drug_max,
                    'cell_line': row['cell_line'],
                    'synergy_zip': row['synergy_zip']
                }
                batch_metadata.append(metadata_dict)
                
                # Also add to compact rows with calculated bits
                compact_row = {
                    'drug_min': drug_min,
                    'drug_max': drug_max,
                    'cell_line': row['cell_line'],
                    'synergy_zip': float(row['synergy_zip']),
                    'fp_min_active_bits': int(np.sum(fp_min)),
                    'fp_max_active_bits': int(np.sum(fp_max))
                }
                all_rows.append(compact_row)
        
        # Convert batch to numpy arrays
        if batch_X_fp_min:  # Check if not empty
            try:
                # Ensure consistent dtype
                batch_X_fp_min_arr = np.vstack(batch_X_fp_min).astype(np.float32)
                batch_X_fp_max_arr = np.vstack(batch_X_fp_max).astype(np.float32)
                batch_y_arr = np.array(batch_y_synergy, dtype=np.float32)
                
                # Save as separate batch files
                batch_prefix = f"batch_{batch_idx:04d}_"
                np.save(os.path.join(batch_dir, f"{batch_prefix}X_fp_min.npy"), batch_X_fp_min_arr)
                np.save(os.path.join(batch_dir, f"{batch_prefix}X_fp_max.npy"), batch_X_fp_max_arr)
                np.save(os.path.join(batch_dir, f"{batch_prefix}y_synergy.npy"), batch_y_arr)
                
                # Write metadata incrementally
                append_to_csv(metadata_file, batch_metadata)
                
                total_processed += len(batch_metadata)
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                
            # Clear batch arrays to save memory
            del batch_X_fp_min_arr, batch_X_fp_max_arr, batch_y_arr
        
        # Clear memory after each batch
        del batch_df, batch_X_fp_min, batch_X_fp_max, batch_y_synergy, batch_metadata
        gc.collect()
        
        # Periodically save compact data to avoid memory issues
        if len(all_rows) >= 10000:
            append_to_compact_csv(os.path.join(output_dir, "synergy_with_fingerprints_compact.csv"), all_rows)
            all_rows = []
            gc.collect()
    
    # Save any remaining compact data
    if all_rows:
        append_to_compact_csv(os.path.join(output_dir, "synergy_with_fingerprints_compact.csv"), all_rows)
    
    # Save skipped pairs for analysis
    skipped_df = skipped_synergy_df[['drug_min', 'drug_max', 'synergy_zip']].copy()
    skipped_df['has_fp_min'] = skipped_df['drug_min'].isin(drug_to_fingerprint)
    skipped_df['has_fp_max'] = skipped_df['drug_max'].isin(drug_to_fingerprint)
    
    skipped_path = os.path.join(output_dir, "skipped_pairs.csv")
    skipped_df.to_csv(skipped_path, index=False)
    print(f"Saved information about skipped pairs to {skipped_path}")
    
    print(f"Successfully processed {total_processed} drug combinations")
    print("Done!")
    
    print("\nTo combine the batch files into single arrays, run the combine_batches() function")
    
    return None  # Don't return data to save memory

def combine_batches(output_dir="C:/Users/46762/VSCODE/BIG_PHARMA/data/interim/merged_data"):
    """Combine batch files into single arrays"""
    print("Combining batch files into single arrays...")
    
    np_dir = os.path.join(output_dir, "numpy_arrays")
    batch_dir = os.path.join(np_dir, "batches")
    
    # Get lists of batch files
    fp_min_files = sorted(glob.glob(os.path.join(batch_dir, "*X_fp_min.npy")))
    fp_max_files = sorted(glob.glob(os.path.join(batch_dir, "*X_fp_max.npy")))
    y_files = sorted(glob.glob(os.path.join(batch_dir, "*y_synergy.npy")))
    
    print(f"Found {len(fp_min_files)} batch files for X_fp_min")
    print(f"Found {len(fp_max_files)} batch files for X_fp_max")
    print(f"Found {len(y_files)} batch files for y_synergy")
    
    if not fp_min_files or not fp_max_files or not y_files:
        print("No batch files found. Run merge_fingerprints_with_synergy first.")
        return
        
    # Combine X_fp_min files
    try:
        print("Combining X_fp_min files...")
        combined_fp_min = []
        
        for i, file in enumerate(tqdm(fp_min_files)):
            try:
                arr = np.load(file)
                combined_fp_min.append(arr)
                del arr
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        X_fp_min = np.vstack(combined_fp_min).astype(np.float32)
        np.save(os.path.join(np_dir, "X_fp_min.npy"), X_fp_min)
        del combined_fp_min, X_fp_min
        gc.collect()
        print("Combined X_fp_min saved.")
    except Exception as e:
        print(f"Error combining X_fp_min files: {e}")
    
    # Combine X_fp_max files
    try:
        print("Combining X_fp_max files...")
        combined_fp_max = []
        
        for i, file in enumerate(tqdm(fp_max_files)):
            try:
                arr = np.load(file)
                combined_fp_max.append(arr)
                del arr
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        X_fp_max = np.vstack(combined_fp_max).astype(np.float32)
        np.save(os.path.join(np_dir, "X_fp_max.npy"), X_fp_max)
        del combined_fp_max, X_fp_max
        gc.collect()
        print("Combined X_fp_max saved.")
    except Exception as e:
        print(f"Error combining X_fp_max files: {e}")
    
    # Combine y_synergy files
    try:
        print("Combining y_synergy files...")
        combined_y = []
        
        for i, file in enumerate(tqdm(y_files)):
            try:
                arr = np.load(file)
                combined_y.append(arr)
                del arr
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        y_synergy = np.concatenate(combined_y).astype(np.float32)
        np.save(os.path.join(np_dir, "y_synergy_zip.npy"), y_synergy)
        del combined_y, y_synergy
        gc.collect()
        print("Combined y_synergy_zip saved.")
    except Exception as e:
        print(f"Error combining y_synergy files: {e}")
    
    print("Batch combination complete!")

def append_to_csv(file_path, rows):
    """Append rows to a CSV file"""
    if not rows:
        return
        
    # Convert to dataframe
    df = pd.DataFrame(rows)
    
    # Append to file
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)

def append_to_compact_csv(file_path, rows):
    """Append rows to compact CSV file"""
    if not rows:
        return
    
    # Convert to dataframe
    df = pd.DataFrame(rows)
    
    # Append to file
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)

if __name__ == "__main__":
    # Step 1: Process and save data in batches
    merge_fingerprints_with_synergy(batch_size=250, max_pairs=None)  # Set max_pairs=None for all pairs
    
    # Step 2: Combine batches into single arrays
    # Uncomment if you want to run this step now
    # combine_batches()
    
    print("\nProcess completed!")
    print("First run generated the compact CSV and batch files.")
    print("Run combine_batches() separately to merge the batch files when needed.")


"""from merge_fingerprint import combine_batches
combine_batches()
# Load the NumPy arrays
X_fp_min = np.load("data/interim/merged_data/numpy_arrays/X_fp_min.npy")
X_fp_max = np.load("data/interim/merged_data/numpy_arrays/X_fp_max.npy")
y_synergy = np.load("data/interim/merged_data/numpy_arrays/y_synergy_zip.npy")

# Load metadata for reference
metadata = pd.read_csv("data/interim/merged_data/numpy_arrays/metadata.csv")"""