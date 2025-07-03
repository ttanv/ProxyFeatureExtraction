"""
Tests for data io file
"""

import pytest
import pandas as pd
from pathlib import Path
import shutil
from feature_extraction.data_io import DataIO

@pytest.fixture
def test_data_dir(tmp_path):
    # Create test directory structure
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    
    # Create test folders
    folders = ["folder1", "folder2", "folder3"]
    for folder in folders:
        folder_path = data_dir / folder
        folder_path.mkdir()
        
        # Create test CSV files in each folder
        sample_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        sample_data.to_csv(folder_path / "proxy_conn.csv", index=False)
        sample_data.to_csv(folder_path / "background_conn_labeled.csv", index=False)
        sample_data.to_csv(folder_path / "relayed_conn_labeled.csv", index=False)
    
    return data_dir

@pytest.fixture
def test_split_csv(tmp_path):
    # Create test split CSV
    split_data = pd.DataFrame({
        'folder_name': ['folder1', 'folder2'],
        'split': ['train', 'train']
    })
    csv_path = tmp_path / "split.csv"
    split_data.to_csv(csv_path, index=False)
    return csv_path

def test_get_folders_without_csv(test_data_dir):
    output_dir = Path("/tmp/output")
    data_io = DataIO(test_data_dir, None, None, output_dir)
    folders = data_io.folder_paths
    
    assert len(folders) == 3
    assert all(folder.is_dir() for folder in folders)

def test_get_folders_with_csv(test_data_dir, test_split_csv):
    output_dir = Path("/tmp/output")
    data_io = DataIO(test_data_dir, test_split_csv, "train", output_dir)
    folders = data_io.folder_paths
    
    assert len(folders) == 2
    assert all(folder.name in ['folder1', 'folder2'] for folder in folders)

def test_load_gateway_batches(test_data_dir):
    output_dir = Path("/tmp/output")
    data_io = DataIO(test_data_dir, None, None, output_dir)
    
    batch_size = 2
    batches = list(data_io.load_gateway_batches(batch_size))
    
    assert len(batches) == 2  # Should have 2 batches (2 folders + 1 folder)
    assert len(batches[0]) == 2  # First batch should have 2 DataFrames
    assert len(batches[1]) == 1  # Second batch should have 1 DataFrame

def test_save_bg_batch(test_data_dir):
    output_dir = Path("/tmp/test_output")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    data_io = DataIO(test_data_dir, None, None, output_dir)
    test_dfs = [pd.DataFrame({'test': [1, 2, 3]})] * 2
    
    data_io.save_bg_batch(test_dfs, "test_features", 1)
    
    output_file = output_dir / "bg_test_features_batch_1.csv"
    assert output_file.exists()
    saved_df = pd.read_csv(output_file)
    assert len(saved_df) == 6  # 2 DataFrames with 3 rows each

def test_load_df_not_found(test_data_dir):
    output_dir = Path("/tmp/output")
    data_io = DataIO(test_data_dir, None, None, output_dir)
    
    with pytest.raises(FileNotFoundError):
        data_io._load_df(test_data_dir / "nonexistent.csv")

def test_load_df_empty(test_data_dir):
    empty_file = test_data_dir / "empty.csv"
    empty_file.touch()
    
    output_dir = Path("/tmp/output")
    data_io = DataIO(test_data_dir, None, None, output_dir)
    
    with pytest.raises(ValueError):
        data_io._load_df(empty_file)


