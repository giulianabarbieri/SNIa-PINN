import os
import pandas as pd
# pyrefly: ignore [missing-import]
from alerce.core import Alerce
from tqdm import tqdm

class DataIngestor:
    """
    Handles data acquisition from the ALeRCE broker API.
    Facilitates the download of light curves for transient classification.
    """
    
    def __init__(self, output_dir="dataset"):
        self.client = Alerce()
        self.output_dir = output_dir
        self.metadata_file = os.path.join(self.output_dir, "metadata.csv")
        
        # Ensure the data directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize metadata file if it doesn't exist
        if not os.path.exists(self.metadata_file):
            pd.DataFrame(columns=['oid', 'class_name']).to_csv(self.metadata_file, index=False)

    def fetch_sample_targets(self, class_name, count=50):
        """
        Retrieves a list of objects based on their astronomical classification.
        Es el que trae los objetos del broker.
        """
        print(f"[*] Fetching {count} targets for class: {class_name}...")
        
        targets = self.client.query_objects(
            classifier="lc_classifier_BHRF_forced_phot",
            class_name=class_name,
            probability=0.7,
            page_size=count,
            format="pandas"
        )
        
        # ALeRCE sometimes uses 'oid' (Object ID) or 'aid' (ALeRCE ID)
        # We ensure we have a column named 'oid' for the rest of the script
        if 'oid' not in targets.columns:
            if 'aid' in targets.columns:
                targets = targets.rename(columns={'aid': 'oid'})
            else:
                # If it's in the index, move it to a column
                targets = targets.reset_index()
                # If after reset it's called 'index' or 'aid', rename it
                if 'oid' not in targets.columns:
                    targets.rename(columns={targets.columns[0]: 'oid'}, inplace=True)
            
        return targets

    def download_and_save(self, oid, class_name=None):
        """
        Saves the resulting dataframe to a CSV file.
        """
        try:
            # Query detections (flux vs time)
            detections = self.client.query_detections(oid, format="pandas")

            # Save to disk for local processing
            file_path = os.path.join(self.output_dir, f"{oid}_detections.csv")
            detections.to_csv(file_path, index=False)
            return True
        except Exception as e:
            print(f"[!] Error downloading {oid}: {e}")
            return False

if __name__ == "__main__":
    ingestor = DataIngestor()
    samples_per_class = 2000
    
    targets = ingestor.fetch_sample_targets("SNIa", count=samples_per_class)

    if not targets.empty:
        print(f"\nDescargando datos")
        for oid in tqdm(targets['oid'], desc=f"Progreso", unit="obj"):
            # Tomamos los objetos del broker y los guardamos en archivos CSV
            success = ingestor.download_and_save(oid, class_name="SNIa")
            
            # Optional: log without breaking the progress bar
            if success:
                tqdm.write(f"[+] Finalizado: {oid}") 
    else:
        print(f"[!] No targets found for class: SNIa")