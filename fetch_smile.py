#%%
import pandas as pd
import requests
import time
from tqdm import tqdm
import os
import json
import random

def get_smile_from_pubchem(cid, cache=None):
    """
    Fetch SMILE string from PubChem using a compound ID (CID)
    With caching to avoid redundant API calls
    """
    if pd.isna(cid) or cid == "":
        return None
    
    # Convert float CID to integer (remove .0)
    try:
        cid_int = str(int(float(cid)))
    except:
        cid_int = str(cid)
    
    # Check cache first if provided
    if cache is not None and cid_int in cache:
        print(f"✓ Found SMILE in cache for CID {cid_int}: {cache[cid_int][:30]}{'...' if len(cache[cid_int]) > 30 else ''}")
        return cache[cid_int]
    
    # First try to get canonical SMILES
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_int}/property/IsomericSMILES,CanonicalSMILES,ConnectivitySMILES/JSON"
    print(f"API request: {url}")
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                properties = data['PropertyTable']['Properties'][0]
                
                # Try to get SMILES in order of preference
                smile = None
                if 'IsomericSMILES' in properties:
                    smile = properties['IsomericSMILES']
                elif 'CanonicalSMILES' in properties:
                    smile = properties['CanonicalSMILES']
                elif 'ConnectivitySMILES' in properties:
                    smile = properties['ConnectivitySMILES']
                
                if smile:
                    # Update cache if provided
                    if cache is not None:
                        cache[cid_int] = smile
                    print(f"✓ Found SMILE for CID {cid_int}: {smile[:30]}{'...' if len(smile) > 30 else ''}")
                    return smile
        
        # If we get here, we didn't find SMILES or the request failed
        print(f"✘ Failed to retrieve SMILE for CID {cid_int} (Status code: {response.status_code})")
        
        # Try fallback method - get structure directly
        fallback_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_int}/record/SDF/?record_type=2d&response_type=display"
        print(f"Trying fallback method: {fallback_url}")
        
        fallback_response = requests.get(fallback_url)
        if fallback_response.status_code == 200:
            # Extract SMILES from SDF
            sdf_content = fallback_response.text
            lines = sdf_content.split('\n')
            for i, line in enumerate(lines):
                if '>  <PUBCHEM_OPENEYE_CAN_SMILES>' in line and i+1 < len(lines):
                    smile = lines[i+1].strip()
                    if smile:
                        # Update cache if provided
                        if cache is not None:
                            cache[cid_int] = smile
                        print(f"✓ Found SMILE for CID {cid_int} (fallback): {smile[:30]}{'...' if len(smile) > 30 else ''}")
                        return smile
        
        return None
    except Exception as e:
        print(f"✘ Error fetching SMILE for CID {cid_int}: {e}")
        return None

def load_cache(cache_path):
    """Load cached SMILE strings from file"""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except:
            print("Error loading cache file. Creating new cache.")
    return {}

def save_cache(cache, cache_path):
    """Save cached SMILE strings to file"""
    with open(cache_path, 'w') as f:
        json.dump(cache, f)
    print(f"Cache saved with {len(cache)} entries")

def fetch_smiles_for_subset(n=10, random_selection=True):
    """
    Process only a subset of drugs (for testing)
    
    Parameters:
    n (int): Number of drugs to process
    random_selection (bool): If True, select random drugs; if False, select first n
    """
    # Define the paths
    csv_path = "C:/Users/46762/VSCODE/BIG_PHARMA/data/interim/drug_cids.csv"
    output_path = "C:/Users/46762/VSCODE/BIG_PHARMA/data/interim/drug_smiles_subset.csv"
    cache_path = "C:/Users/46762/VSCODE/BIG_PHARMA/data/interim/smile_cache.json"
    
    # Load cache
    cache = load_cache(cache_path)
    print(f"Loaded cache with {len(cache)} entries")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: File not found - {csv_path}")
        return
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get unique drug-CID pairs, excluding those without CIDs
    unique_drugs = df.dropna(subset=['cid']).drop_duplicates(['drug', 'cid'])
    print(f"Found {len(unique_drugs)} unique drugs with CIDs")
    
    # Select subset
    if random_selection:
        if n > len(unique_drugs):
            print(f"Warning: Requested {n} drugs but only {len(unique_drugs)} available")
            n = len(unique_drugs)
        subset = unique_drugs.sample(n=n)
        print(f"Randomly selected {n} drugs for processing")
    else:
        subset = unique_drugs.head(n)
        print(f"Selected first {n} drugs for processing")
    
    # Create a new DataFrame for results
    results = []
    
    # Fetch SMILE strings for each drug in the subset
    for idx, row in tqdm(subset.iterrows(), total=len(subset), desc="Fetching SMILE strings"):
        drug_name = row['drug']
        cid = row['cid']
        
        # Skip if CID is missing
        if pd.isna(cid) or cid == "":
            print(f"Skipping {drug_name}: No CID available")
            continue
            
        # Get SMILE
        print(f"\nProcessing {drug_name} (CID: {cid})")
        smile = get_smile_from_pubchem(cid, cache=cache)
        
        # Store results
        if smile:
            results.append({
                'drug': drug_name,
                'cid': int(float(cid)),  # Convert to clean integer
                'smile': smile
            })
        
        # Be kind to the PubChem API with a small delay between requests
        time.sleep(0.3)
    
    # Save cache
    save_cache(cache, cache_path)
    
    # Create DataFrame and save results
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    else:
        print("No results to save")
    
    # Report summary
    print(f"\n--- Summary ---")
    print(f"Successfully fetched SMILE strings for {len(results)} out of {len(subset)} drugs")
    if len(results) < len(subset):
        print(f"Could not fetch SMILE strings for {len(subset) - len(results)} drugs")

def fetch_smiles_for_drugs():
    """
    Main function to fetch SMILE strings for all unique drugs with CIDs
    """
    # Define the paths
    csv_path = "C:/Users/46762/VSCODE/BIG_PHARMA/data/interim/drug_cids.csv"
    output_path = "C:/Users/46762/VSCODE/BIG_PHARMA/data/interim/drug_smiles.csv"
    cache_path = "C:/Users/46762/VSCODE/BIG_PHARMA/data/interim/smile_cache.json"
    
    # Load cache
    cache = load_cache(cache_path)
    print(f"Loaded cache with {len(cache)} entries")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: File not found - {csv_path}")
        return
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get unique drug-CID pairs, excluding those without CIDs
    unique_drugs = df.dropna(subset=['cid']).drop_duplicates(['drug', 'cid'])
    print(f"Found {len(unique_drugs)} unique drugs with CIDs")
    
    # Create a new DataFrame for results
    results = []
    
    # Fetch SMILE strings for each drug
    for idx, row in tqdm(unique_drugs.iterrows(), total=len(unique_drugs), desc="Fetching SMILE strings"):
        drug_name = row['drug']
        cid = row['cid']
        
        # Skip if CID is missing
        if pd.isna(cid) or cid == "":
            continue
            
        # Get SMILE
        smile = get_smile_from_pubchem(cid, cache=cache)
        
        # Store results
        if smile:
            results.append({
                'drug': drug_name,
                'cid': int(float(cid)),  # Convert to clean integer
                'smile': smile
            })
        
        # Save cache periodically (every 50 drugs)
        if idx % 50 == 0 and idx > 0:
            save_cache(cache, cache_path)
            
        # Be kind to the PubChem API with a small delay between requests
        time.sleep(0.3)
    
    # Final cache save
    save_cache(cache, cache_path)
    
    # Create DataFrame and save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    
    # Print summary
    print(f"Successfully fetched SMILE strings for {len(results)} drugs")
    print(f"Results saved to {output_path}")
    
    # Report drugs without SMILE strings
    drugs_with_cids = len(unique_drugs)
    drugs_with_smiles = len(results)
    print(f"Could not fetch SMILE strings for {drugs_with_cids - drugs_with_smiles} drugs with CIDs")

if __name__ == "__main__":
    # Run the subset function for testing with 10 random drugs
    # Uncomment the line below to test with a small subset
    #fetch_smiles_for_subset(n=10, random_selection=True)
    
    # Uncomment the line below to process all drugs
     fetch_smiles_for_drugs()