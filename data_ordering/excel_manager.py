"""
Excel/CSV manager for the data ordering tool.
Handles all registry files (annotations, text files, other files, hashes).
"""
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
import pandas as pd


@dataclass
class ImageRecord:
    """Record for an image in the main registry."""
    uuid: str
    specimen_id: Optional[str] = None
    original_path: Optional[str] = None
    current_path: Optional[str] = None
    macroclass_label: Optional[str] = None
    class_label: Optional[str] = None
    genera_label: Optional[str] = None
    fecha_captura: Optional[str] = None  # YYYYMMDD format
    campaign_year: Optional[str] = None
    fuente: Optional[str] = None  # MUPA, YCLH, etc.
    comentarios: Optional[str] = None
    hash_perceptual: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @staticmethod
    def generate_uuid() -> str:
        """Generate a new UUID for an image."""
        return str(uuid.uuid4())


@dataclass
class TextFileRecord:
    """Record for a text file that may contain annotations."""
    id: str
    original_path: str
    current_path: str
    original_filename: str
    file_type: str  # extension
    processed: bool = False
    extracted_info: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass 
class OtherFileRecord:
    """Record for other (non-image, non-text) files."""
    id: str
    original_path: str
    current_path: str
    original_filename: str
    file_type: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class HashRecord:
    """Record for image hashes used in deduplication."""
    uuid: str
    hash_value: str
    file_path: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ExcelManager:
    """
    Manages Excel/CSV registry files for the data ordering process.
    """
    
    def __init__(self, registries_dir: Path):
        """
        Initialize the Excel manager.
        
        Args:
            registries_dir: Directory where registry files are stored
        """
        self.registries_dir = Path(registries_dir)
        self.registries_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.main_registry_path = self.registries_dir / "anotaciones.xlsx"
        self.text_files_path = self.registries_dir / "archivos_texto.xlsx"
        self.other_files_path = self.registries_dir / "archivos_otros.xlsx"
        self.hash_registry_path = self.registries_dir / "hashes.xlsx"
        
        # In-memory dataframes (loaded on demand)
        self._main_df: Optional[pd.DataFrame] = None
        self._text_df: Optional[pd.DataFrame] = None
        self._other_df: Optional[pd.DataFrame] = None
        self._hash_df: Optional[pd.DataFrame] = None
        
        # UUID counter for text/other files
        self._text_counter = 0
        self._other_counter = 0
    
    # === Main Registry (Images) ===
    
    def _get_main_df(self) -> pd.DataFrame:
        """Load or create the main registry dataframe."""
        if self._main_df is None:
            if self.main_registry_path.exists():
                self._main_df = pd.read_excel(self.main_registry_path)
            else:
                # Create empty dataframe with all columns
                columns = list(ImageRecord.__dataclass_fields__.keys())
                self._main_df = pd.DataFrame(columns=columns)
        return self._main_df
    
    def add_image(self, record: ImageRecord) -> str:
        """
        Add a new image record to the main registry.
        
        Args:
            record: The ImageRecord to add
            
        Returns:
            The UUID of the added record
        """
        df = self._get_main_df()
        new_row = pd.DataFrame([asdict(record)])
        self._main_df = pd.concat([df, new_row], ignore_index=True)
        return record.uuid
    
    def get_image_by_uuid(self, uuid: str) -> Optional[Dict]:
        """Get an image record by UUID."""
        df = self._get_main_df()
        matches = df[df['uuid'] == uuid]
        if len(matches) > 0:
            return matches.iloc[0].to_dict()
        return None
    
    def update_image(self, uuid: str, updates: Dict[str, Any]) -> bool:
        """
        Update an image record.
        
        Args:
            uuid: The UUID of the record to update
            updates: Dictionary of field: value to update
            
        Returns:
            True if record was found and updated
        """
        df = self._get_main_df()
        mask = df['uuid'] == uuid
        if mask.any():
            for key, value in updates.items():
                if key in df.columns:
                    df.loc[mask, key] = value
            return True
        return False
    
    def find_images(self, **criteria) -> pd.DataFrame:
        """
        Find images matching criteria.
        
        Args:
            **criteria: Field=value pairs to match
            
        Returns:
            DataFrame of matching records
        """
        df = self._get_main_df()
        mask = pd.Series([True] * len(df))
        for key, value in criteria.items():
            if key in df.columns:
                mask &= (df[key] == value)
        return df[mask]
    
    # === Text Files Registry ===
    
    def _get_text_df(self) -> pd.DataFrame:
        """Load or create the text files registry."""
        if self._text_df is None:
            if self.text_files_path.exists():
                self._text_df = pd.read_excel(self.text_files_path)
                # Update counter
                if len(self._text_df) > 0:
                    max_id = self._text_df['id'].str.extract(r'TXT(\d+)')[0].astype(float).max()
                    self._text_counter = int(max_id) if pd.notna(max_id) else 0
            else:
                columns = list(TextFileRecord.__dataclass_fields__.keys())
                self._text_df = pd.DataFrame(columns=columns)
        return self._text_df
    
    def add_text_file(self, original_path: Path, current_path: Path) -> str:
        """
        Add a text file to the registry.
        
        Returns:
            The generated ID
        """
        df = self._get_text_df()
        self._text_counter += 1
        file_id = f"TXT{self._text_counter:06d}"
        
        record = TextFileRecord(
            id=file_id,
            original_path=str(original_path),
            current_path=str(current_path),
            original_filename=original_path.name,
            file_type=original_path.suffix.lower()
        )
        
        new_row = pd.DataFrame([asdict(record)])
        self._text_df = pd.concat([df, new_row], ignore_index=True)
        return file_id
    
    # === Other Files Registry ===
    
    def _get_other_df(self) -> pd.DataFrame:
        """Load or create the other files registry."""
        if self._other_df is None:
            if self.other_files_path.exists():
                self._other_df = pd.read_excel(self.other_files_path)
                if len(self._other_df) > 0:
                    max_id = self._other_df['id'].str.extract(r'OTH(\d+)')[0].astype(float).max()
                    self._other_counter = int(max_id) if pd.notna(max_id) else 0
            else:
                columns = list(OtherFileRecord.__dataclass_fields__.keys())
                self._other_df = pd.DataFrame(columns=columns)
        return self._other_df
    
    def add_other_file(self, original_path: Path, current_path: Path) -> str:
        """
        Add an 'other' file to the registry.
        
        Returns:
            The generated ID
        """
        df = self._get_other_df()
        self._other_counter += 1
        file_id = f"OTH{self._other_counter:06d}"
        
        record = OtherFileRecord(
            id=file_id,
            original_path=str(original_path),
            current_path=str(current_path),
            original_filename=original_path.name,
            file_type=original_path.suffix.lower()
        )
        
        new_row = pd.DataFrame([asdict(record)])
        self._other_df = pd.concat([df, new_row], ignore_index=True)
        return file_id
    
    # === Hash Registry ===
    
    def _get_hash_df(self) -> pd.DataFrame:
        """Load or create the hash registry."""
        if self._hash_df is None:
            if self.hash_registry_path.exists():
                self._hash_df = pd.read_excel(self.hash_registry_path)
            else:
                columns = list(HashRecord.__dataclass_fields__.keys())
                self._hash_df = pd.DataFrame(columns=columns)
        return self._hash_df
    
    def add_hash(self, uuid: str, hash_value: str, file_path: Path):
        """Add a hash record."""
        df = self._get_hash_df()
        record = HashRecord(uuid=uuid, hash_value=hash_value, file_path=str(file_path))
        new_row = pd.DataFrame([asdict(record)])
        self._hash_df = pd.concat([df, new_row], ignore_index=True)
    
    def find_by_hash(self, hash_value: str) -> Optional[Dict]:
        """Find a record by hash value."""
        df = self._get_hash_df()
        matches = df[df['hash_value'] == hash_value]
        if len(matches) > 0:
            return matches.iloc[0].to_dict()
        return None
    
    def get_all_hashes(self) -> Dict[str, str]:
        """Get all hashes as {hash: uuid} dictionary."""
        df = self._get_hash_df()
        return dict(zip(df['hash_value'], df['uuid']))
    
    # === Save/Load ===
    
    def save_all(self):
        """Save all registries to disk."""
        if self._main_df is not None:
            self._main_df.to_excel(self.main_registry_path, index=False)
        if self._text_df is not None:
            self._text_df.to_excel(self.text_files_path, index=False)
        if self._other_df is not None:
            self._other_df.to_excel(self.other_files_path, index=False)
        if self._hash_df is not None:
            self._hash_df.to_excel(self.hash_registry_path, index=False)
    
    def get_stats(self) -> Dict[str, int]:
        """Get counts of records in each registry."""
        return {
            'images': len(self._get_main_df()),
            'text_files': len(self._get_text_df()),
            'other_files': len(self._get_other_df()),
            'hashes': len(self._get_hash_df()),
        }


# Quick test function
def test_excel_manager():
    """Quick test of the Excel manager."""
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ExcelManager(Path(tmpdir))
        
        # Test adding images
        record1 = ImageRecord(
            uuid=ImageRecord.generate_uuid(),
            specimen_id="12345678",
            original_path="/path/to/image1.jpg",
            macroclass_label="Arthropoda",
            fuente="MUPA",
            comentarios="[AUTO] specimen_id from filename"
        )
        uuid1 = manager.add_image(record1)
        
        record2 = ImageRecord(
            uuid=ImageRecord.generate_uuid(),
            specimen_id="87654321",
            original_path="/path/to/image2.jpg",
            macroclass_label="Botany",
            fuente="YCLH"
        )
        manager.add_image(record2)
        
        # Test retrieval
        retrieved = manager.get_image_by_uuid(uuid1)
        assert retrieved is not None
        assert retrieved['specimen_id'] == "12345678"
        
        # Test update
        manager.update_image(uuid1, {'class_label': 'Insecta'})
        updated = manager.get_image_by_uuid(uuid1)
        assert updated['class_label'] == 'Insecta'
        
        # Test find
        mupa_images = manager.find_images(fuente="MUPA")
        assert len(mupa_images) == 1
        
        # Test text file
        text_id = manager.add_text_file(
            Path("/original/notes.txt"),
            Path("/dest/TXT000001_notes.txt")
        )
        assert text_id == "TXT000001"
        
        # Test hash
        manager.add_hash(uuid1, "abc123hash", Path("/path/to/image1.jpg"))
        found = manager.find_by_hash("abc123hash")
        assert found is not None
        assert found['uuid'] == uuid1
        
        # Test save
        manager.save_all()
        
        # Verify files exist
        assert manager.main_registry_path.exists()
        assert manager.text_files_path.exists()
        assert manager.hash_registry_path.exists()
        
        # Test stats
        stats = manager.get_stats()
        assert stats['images'] == 2
        assert stats['text_files'] == 1
        
        print("✓ Excel manager test passed!")
        print(f"Stats: {stats}")


if __name__ == "__main__":
    test_excel_manager()
