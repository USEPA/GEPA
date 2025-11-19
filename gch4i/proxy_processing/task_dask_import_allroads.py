"""
Name:                   task_dask_import_allroads.py
Date Last Modified:     2025-09-29
Authors Name:           A. Burnette (RTI International)
Purpose:                Import all roads data from the US Census Bureau
Input Files:            -
Output Files:           - tl_{year}_us_allroads.parquet
Notes:                  - Script uses dask to mitigate computational load
                        - Writes out parquet files for each year to raw_roads directory
                        - Includes comprehensive retry logic for failed downloads
"""

from pathlib import Path
from typing import Annotated, List, Tuple
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from dask import delayed, compute
import geopandas as gpd
import pandas as pd
from zipfile import ZipFile
import io
import tempfile
import os
import time
import json
from datetime import datetime
import pickle

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from gch4i.config import (
    V4_DATA_PATH,
    years
)

# TESTING
years = [2023]

from pytask import Product, mark, task

raw_roads_path = Path(V4_DATA_PATH) / "open_data" / "mobile_combustion" / "raw" / "raw_roads"
census_geometry_list = ["roads"]

# Progress tracking files
PROGRESS_DIR = raw_roads_path / "progress"
FAILED_URLS_FILE = PROGRESS_DIR / "failed_urls.json"
SUCCESS_CACHE_FILE = PROGRESS_DIR / "successful_downloads.pkl"
RETRY_LOG_FILE = PROGRESS_DIR / "retry_log.json"

def create_kwargs(geometry_list):
    id_to_kwargs = {}
    for year in years:
        for geometry_type in geometry_list:
            label = f"{str(year)}_{geometry_type}"
            url = f"https://www2.census.gov/geo/tiger/TIGER{year}/{geometry_type.upper()}/"
            output_path = raw_roads_path / f"tl_{year}_us_all{geometry_type.lower()}.parquet"
            id_to_kwargs[label] = {"url": url, "output_path": output_path}
    return id_to_kwargs

def setup_progress_tracking():
    """Initialize progress tracking directory and files"""
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not FAILED_URLS_FILE.exists():
        with open(FAILED_URLS_FILE, 'w') as f:
            json.dump([], f)
    
    if not SUCCESS_CACHE_FILE.exists():
        with open(SUCCESS_CACHE_FILE, 'wb') as f:
            pickle.dump({}, f)
    
    if not RETRY_LOG_FILE.exists():
        with open(RETRY_LOG_FILE, 'w') as f:
            json.dump([], f)

def load_progress():
    """Load previous progress"""
    setup_progress_tracking()
    
    with open(FAILED_URLS_FILE, 'r') as f:
        failed_urls = json.load(f)
    
    with open(SUCCESS_CACHE_FILE, 'rb') as f:
        success_cache = pickle.load(f)
    
    return failed_urls, success_cache

def save_progress(failed_urls, success_cache):
    """Save current progress"""
    with open(FAILED_URLS_FILE, 'w') as f:
        json.dump(failed_urls, f, indent=2)
    
    with open(SUCCESS_CACHE_FILE, 'wb') as f:
        pickle.dump(success_cache, f)

def log_retry_attempt(zip_url, attempt, error_msg):
    """Log retry attempts for debugging"""
    log_entry = {
        "url": zip_url,
        "attempt": attempt,
        "timestamp": datetime.now().isoformat(),
        "error": str(error_msg)
    }
    
    try:
        with open(RETRY_LOG_FILE, 'r') as f:
            log_data = json.load(f)
    except:
        log_data = []
    
    log_data.append(log_entry)
    
    # Keep only last 1000 entries to prevent file from getting too large
    if len(log_data) > 1000:
        log_data = log_data[-1000:]
    
    with open(RETRY_LOG_FILE, 'w') as f:
        json.dump(log_data, f, indent=2)

# Set up robust session with retries
def create_robust_session():
    retry_strategy = Retry(
        total=8,
        backoff_factor=2,
        status_forcelist=[408, 429, 500, 502, 503, 504, 520, 522, 524],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=10
    )
    
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    return session

session = create_robust_session()

# Updated headers
hdr = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'en-US,en;q=0.8',
    'Connection': 'keep-alive'
}

def download_with_comprehensive_retry(zip_url, valid_mtfcc, state_fips_code, success_cache):
    """
    Download with multiple retry strategies and caching
    """
    # Check if we already have this file successfully processed
    if zip_url in success_cache:
        print(f"✓ Using cached result for {zip_url}")
        return success_cache[zip_url]
    
    max_attempts = 5
    backoff_times = [1, 2, 5, 10, 20]  # Progressive backoff
    
    for attempt in range(max_attempts):
        try:
            print(f"Downloading {zip_url} (attempt {attempt + 1}/{max_attempts})")
            
            # Progressive timeout increase
            timeout = min(60 + (attempt * 30), 300)  # 60s to 300s max
            
            with session.get(
                zip_url, 
                stream=True, 
                headers=hdr, 
                timeout=timeout
            ) as response:
                response.raise_for_status()
                
                # Validate response
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) < 1000:
                    raise ValueError(f"File too small: {content_length} bytes")
                
                # Download in chunks to handle large files better
                content_chunks = []
                total_size = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        content_chunks.append(chunk)
                        total_size += len(chunk)
                
                content = b''.join(content_chunks)
                
                if total_size < 1000:
                    raise ValueError(f"Downloaded content too small: {total_size} bytes")
                
                # Process the zip file
                with ZipFile(io.BytesIO(content)) as zip_file:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        zip_file.extractall(path=temp_dir)
                        shp_file = next(
                            (os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".shp")), 
                            None
                        )
                        
                        if not shp_file:
                            raise ValueError("No .shp file found in zip")
                        
                        gdf = gpd.read_file(shp_file)
                        if len(gdf) == 0:
                            print(f"Warning: {zip_url} contains no features, but processing as success")
                            return gpd.GeoDataFrame()  # Return empty but valid GeoDataFrame
                        
                        # Fix the SettingWithCopyWarning
                        gdf_filtered = gdf[gdf['MTFCC'].isin(valid_mtfcc)].copy()
                        gdf_filtered.loc[:, 'state_fips'] = state_fips_code
                        
                        print(f"✓ Successfully processed {zip_url}: {len(gdf_filtered)} features")
                        
                        # Cache successful result
                        success_cache[zip_url] = gdf_filtered
                        
                        return gdf_filtered
                        
        except Exception as e:
            error_msg = str(e)
            print(f"Attempt {attempt + 1} failed for {zip_url}: {error_msg}")
            log_retry_attempt(zip_url, attempt + 1, error_msg)
            
            if attempt < max_attempts - 1:
                sleep_time = backoff_times[attempt]
                print(f"Waiting {sleep_time} seconds before retry...")
                time.sleep(sleep_time)
            else:
                print(f"✗ Final failure for {zip_url} after {max_attempts} attempts")
    
    return None

def process_urls_in_batches(zip_urls, valid_mtfcc, batch_size=50):
    """
    Process URLs in batches with retry logic for failed downloads
    """
    failed_urls, success_cache = load_progress()
    all_results = []
    
    # Start with failed URLs from previous runs
    urls_to_process = list(set(failed_urls + zip_urls))
    total_urls = len(urls_to_process)
    
    print(f"Total URLs to process: {total_urls}")
    print(f"Previously successful: {len(success_cache)}")
    print(f"Previously failed: {len(failed_urls)}")
    
    # Process in batches
    for batch_start in range(0, len(urls_to_process), batch_size):
        batch_end = min(batch_start + batch_size, len(urls_to_process))
        batch_urls = urls_to_process[batch_start:batch_end]
        
        print(f"\n--- Processing batch {batch_start//batch_size + 1} ({batch_start+1}-{batch_end} of {total_urls}) ---")
        
        # Create delayed tasks for this batch
        delayed_tasks = []
        for zip_url in batch_urls:
            try:
                state_fips_code = int(zip_url.split("_")[2][:2])
                delayed_task = delayed(download_with_comprehensive_retry)(
                    zip_url, valid_mtfcc, state_fips_code, success_cache
                )
                delayed_tasks.append((zip_url, delayed_task))
            except (ValueError, IndexError):
                print(f"Skipping malformed URL: {zip_url}")
                continue
        
        # Execute batch
        if delayed_tasks:
            batch_urls_list, batch_tasks = zip(*delayed_tasks)
            results = compute(*batch_tasks)
            
            # Process results
            batch_failed = []
            batch_successful = 0
            
            for url, result in zip(batch_urls_list, results):
                if result is not None and isinstance(result, gpd.GeoDataFrame):
                    all_results.append(result)
                    batch_successful += 1
                    # Remove from failed list if it was there
                    if url in failed_urls:
                        failed_urls.remove(url)
                else:
                    batch_failed.append(url)
                    if url not in failed_urls:
                        failed_urls.append(url)
            
            print(f"Batch complete: {batch_successful} successful, {len(batch_failed)} failed")
            
            # Save progress after each batch
            save_progress(failed_urls, success_cache)
            
            # Brief pause between batches to be nice to the server
            if batch_end < len(urls_to_process):
                print("Pausing 10 seconds between batches...")
                time.sleep(10)
    
    return all_results, failed_urls

_ID_TO_DL_KWARGS = create_kwargs(census_geometry_list)

for _id, kwargs in _ID_TO_DL_KWARGS.items():
    @mark.persist
    @task(id=_id, kwargs=kwargs)
    def task_download_census_geo(url: str, output_path: Annotated[Path, Product]):
        print(f"=== Starting Census Roads Download ===")
        print(f"Source URL: {url}")
        print(f"Output: {output_path}")
        
        setup_progress_tracking()
        
        # Get list of files
        with session.get(url, headers=hdr, timeout=30) as r:
            r.raise_for_status()
            soup = BeautifulSoup(r.content, features="html.parser")

        valid_fips = list(range(1, 2)) + list(range(4, 7)) + list(range(8, 14)) + list(range(16, 43)) + list(range(44, 52)) + list(range(53, 57))
        valid_mtfcc = ['S1100', 'S1200', 'S1400', 'S1630', 'S1640']

        # Parse ZIP URLs
        zip_urls = []
        for link in soup.find_all("a"):
            href = link.get("href", "")
            text = link.get_text(strip=True)
            
            if text.endswith(".zip"):
                try:
                    parts = text.split("_")
                    if len(parts) >= 3 and parts[2][:2].isdigit():
                        fips_code = int(parts[2][:2])
                        if fips_code in valid_fips:
                            zip_urls.append(url + href)
                except (ValueError, IndexError):
                    continue

        print(f"Found {len(zip_urls)} zip files to download")
        
        if not zip_urls:
            print("No valid zip files found!")
            return
        
        # Process all URLs with comprehensive retry
        max_retry_rounds = 3
        for retry_round in range(max_retry_rounds):
            print(f"\n=== Download Round {retry_round + 1}/{max_retry_rounds} ===")
            
            all_results, failed_urls = process_urls_in_batches(zip_urls, valid_mtfcc)
            
            print(f"Round {retry_round + 1} complete:")
            print(f"  Successful downloads: {len(all_results)}")
            print(f"  Failed downloads: {len(failed_urls)}")
            
            if not failed_urls:
                print("🎉 All downloads successful!")
                break
            elif retry_round < max_retry_rounds - 1:
                print(f"Retrying {len(failed_urls)} failed downloads in next round...")
                print("Waiting 30 seconds before retry round...")
                time.sleep(30)
        
        # Final processing
        if all_results:
            print(f"\nCombining {len(all_results)} successful downloads...")
            combined_gdf = pd.concat(all_results, ignore_index=True)
            
            if not isinstance(combined_gdf, gpd.GeoDataFrame):
                combined_gdf = gpd.GeoDataFrame(combined_gdf)
            
            combined_gdf.to_parquet(output_path)
            print(f"✓ Successfully wrote {len(combined_gdf)} features to {output_path}")
            print(f"  File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
            
            # Final summary
            total_attempted = len(zip_urls)
            successful = total_attempted - len(failed_urls)
            print(f"\n=== Final Summary ===")
            print(f"Total files attempted: {total_attempted}")
            print(f"Successful downloads: {successful}")
            print(f"Failed downloads: {len(failed_urls)}")
            print(f"Success rate: {successful/total_attempted*100:.1f}%")
            
            if failed_urls:
                print(f"\nFailed URLs saved to: {FAILED_URLS_FILE}")
                print("You can re-run this script to retry failed downloads.")
        else:
            print("✗ No successful downloads. Check network connection and retry.")

# Main execution
if __name__ == "__main__":
    raw_roads_path.mkdir(parents=True, exist_ok=True)
    
    print("=== Census Roads Data Download with Comprehensive Retry ===")
    print(f"Target directory: {raw_roads_path}")
    print(f"Progress tracking: {PROGRESS_DIR}")
    
    for kwargs in _ID_TO_DL_KWARGS.values():
        task_download_census_geo(**kwargs)