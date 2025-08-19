import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator
import dask.dataframe as dd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

from .cache_manager import cache_manager, cached

class DataService:
    """High-performance data service for handling large datasets efficiently."""
    
    def __init__(self, data_dir: str = "data/synthetic"):
        self.data_dir = Path(data_dir)
        self._chunk_size = 10000  # Rows per chunk
        self._max_workers = 4
        self._lock = threading.Lock()
    
    @cached(ttl_seconds=1800, key_prefix="data_summary")
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all datasets."""
        summary = {}
        
        for csv_file in self.data_dir.glob("*.csv"):
            try:
                # Use dask for efficient reading of large files
                ddf = dd.read_csv(csv_file, blocksize="64MB")
                summary[csv_file.stem] = {
                    "file_size_mb": csv_file.stat().st_size / 1024 / 1024,
                    "estimated_rows": len(ddf),
                    "columns": list(ddf.columns),
                    "last_modified": csv_file.stat().st_mtime
                }
            except Exception as e:
                summary[csv_file.stem] = {"error": str(e)}
        
        return summary
    
    @cached(ttl_seconds=3600, key_prefix="fleet_info")
    def load_fleet_info(self) -> pd.DataFrame:
        """Load fleet information efficiently."""
        fleet_file = self.data_dir / "fleet_info.csv"
        if not fleet_file.exists():
            return pd.DataFrame()
        
        return pd.read_csv(fleet_file)
    
    @cached(ttl_seconds=3600, key_prefix="weather")
    def load_weather_data(self) -> pd.DataFrame:
        """Load weather data efficiently."""
        weather_file = self.data_dir / "weather.csv"
        if not weather_file.exists():
            return pd.DataFrame()
        
        return pd.read_csv(weather_file)
    
    def load_segments_chunked(self, chunk_size: int = None) -> Iterator[pd.DataFrame]:
        """Load segments data in chunks to avoid memory issues."""
        segments_file = self.data_dir / "segments.csv"
        if not segments_file.exists():
            return
        
        chunk_size = chunk_size or self._chunk_size
        
        # Use pandas chunked reading
        for chunk in pd.read_csv(segments_file, chunksize=chunk_size):
            yield chunk
    
    @cached(ttl_seconds=1800, key_prefix="segments_sample")
    def load_segments_sample(self, sample_size: int = 10000) -> pd.DataFrame:
        """Load a sample of segments data for quick analysis."""
        segments_file = self.data_dir / "segments.csv"
        if not segments_file.exists():
            return pd.DataFrame()
        
        # Use dask for efficient sampling
        ddf = dd.read_csv(segments_file, blocksize="64MB")
        sample = ddf.sample(frac=min(1.0, sample_size / len(ddf))).compute()
        return sample
    
    def load_routes_efficiently(self, date_filter: str = None) -> pd.DataFrame:
        """Load routes data with optional date filtering."""
        routes_file = self.data_dir / "routes.csv"
        if not routes_file.exists():
            return pd.DataFrame()
        
        if date_filter:
            # Use dask for efficient filtering
            ddf = dd.read_csv(routes_file, blocksize="64MB")
            filtered = ddf[ddf['date'] == date_filter].compute()
            return filtered
        else:
            # Load sample for performance
            return self.load_routes_sample()
    
    @cached(ttl_seconds=1800, key_prefix="routes_sample")
    def load_routes_sample(self, sample_size: int = 5000) -> pd.DataFrame:
        """Load a sample of routes data."""
        routes_file = self.data_dir / "routes.csv"
        if not routes_file.exists():
            return pd.DataFrame()
        
        ddf = dd.read_csv(routes_file, blocksize="64MB")
        sample = ddf.sample(frac=min(1.0, sample_size / len(ddf))).compute()
        return sample
    
    def load_charging_sessions_efficiently(self, date_filter: str = None) -> pd.DataFrame:
        """Load charging sessions with optional filtering."""
        charging_file = self.data_dir / "charging_sessions.csv"
        if not charging_file.exists():
            return pd.DataFrame()
        
        if date_filter:
            ddf = dd.read_csv(charging_file, blocksize="64MB")
            filtered = ddf[ddf['date'] == date_filter].compute()
            return filtered
        else:
            return self.load_charging_sessions_sample()
    
    @cached(ttl_seconds=1800, key_prefix="charging_sample")
    def load_charging_sessions_sample(self, sample_size: int = 3000) -> pd.DataFrame:
        """Load a sample of charging sessions."""
        charging_file = self.data_dir / "charging_sessions.csv"
        if not charging_file.exists():
            return pd.DataFrame()
        
        ddf = dd.read_csv(charging_file, blocksize="64MB")
        sample = ddf.sample(frac=min(1.0, sample_size / len(ddf))).compute()
        return sample
    
    def get_available_dates(self) -> List[str]:
        """Get list of available dates in the dataset."""
        routes_file = self.data_dir / "routes.csv"
        if not routes_file.exists():
            return []
        
        # Use dask for efficient date extraction
        ddf = dd.read_csv(routes_file, blocksize="64MB")
        if 'date' in ddf.columns:
            dates = ddf['date'].unique().compute()
            return sorted(dates.tolist())
        return []
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        summary = self.get_data_summary()
        stats = {
            "total_files": len(summary),
            "total_size_mb": sum(info.get("file_size_mb", 0) for info in summary.values()),
            "available_dates": self.get_available_dates(),
            "file_details": summary
        }
        
        # Add quick metrics
        try:
            fleet_info = self.load_fleet_info()
            stats["fleet_size"] = len(fleet_info)
            
            routes_sample = self.load_routes_sample(1000)
            stats["routes_sample_size"] = len(routes_sample)
            
            charging_sample = self.load_charging_sessions_sample(1000)
            stats["charging_sample_size"] = len(charging_sample)
            
        except Exception as e:
            stats["error"] = str(e)
        
        return stats
    
    def parallel_process_chunks(self, func, data_iterator: Iterator[pd.DataFrame]) -> List[Any]:
        """Process data chunks in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = []
            
            for chunk in data_iterator:
                future = executor.submit(func, chunk)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing chunk: {e}")
        
        return results
    
    def stream_analysis(self, analysis_func, chunk_size: int = 5000) -> Iterator[Dict[str, Any]]:
        """Stream analysis results for large datasets."""
        segments_file = self.data_dir / "segments.csv"
        if not segments_file.exists():
            return
        
        total_rows = sum(1 for _ in open(segments_file)) - 1  # Exclude header
        processed_rows = 0
        
        for chunk in pd.read_csv(segments_file, chunksize=chunk_size):
            try:
                result = analysis_func(chunk)
                processed_rows += len(chunk)
                
                yield {
                    "progress": (processed_rows / total_rows) * 100,
                    "processed_rows": processed_rows,
                    "total_rows": total_rows,
                    "result": result
                }
                
            except Exception as e:
                yield {
                    "error": str(e),
                    "processed_rows": processed_rows,
                    "total_rows": total_rows
                }
    
    def get_memory_efficient_dataframe(self, file_name: str, columns: List[str] = None, 
                                     sample_frac: float = 0.1) -> pd.DataFrame:
        """Get a memory-efficient dataframe with optional column selection and sampling."""
        file_path = self.data_dir / f"{file_name}.csv"
        if not file_path.exists():
            return pd.DataFrame()
        
        try:
            # Use dask for efficient reading
            ddf = dd.read_csv(file_path, blocksize="64MB")
            
            # Select columns if specified
            if columns:
                available_cols = [col for col in columns if col in ddf.columns]
                ddf = ddf[available_cols]
            
            # Sample if requested
            if sample_frac < 1.0:
                ddf = ddf.sample(frac=sample_frac)
            
            # Compute and return
            return ddf.compute()
            
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            return pd.DataFrame()

# Global data service instance
data_service = DataService()
