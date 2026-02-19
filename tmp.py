from pathlib import Path
import pandas as pd

MERGED = Path(r"C:\\merge_4a\\")
regs = MERGED / "registries"
dups_path = MERGED / "Duplicados" / "duplicados_registro.xlsx"

main_df = pd.read_excel(regs / "anotaciones.xlsx") if (regs / "anotaciones.xlsx").exists() else None
hashes_df = pd.read_excel(regs / "hashes.xlsx") if (regs / "hashes.xlsx").exists() else None
dup_df = pd.read_excel(dups_path) if dups_path.exists() else None

# Paste the two lists you showed here (or build them programmatically)
main_only = ['00237a5c','0085f0f8','008f2473','00964038','009f42a7','00cbfc60','0129e280','0145149d','0163737a','01852a78','01931e5d','01936bed','01b65bcb','023ac200','02785714','027ce3fa','02a2479b','02a97d97','02bb7a58','02e80d28']
dup_only  = ['00692d48','00e008f9','029db920-ceee-48f7-bc5f-c1701d5252e1','02ce0080','03e140f1','04ad848c','04cec28f','050a565f-ce21-4aa4-875f-593acbd69299','054b749d','05764c9a','066e1192','06c74e27','07558c1c-0093-4edb-aaa8-f4392826c678','075ac963','082dc105','08300e3e','08971e39','08a9bff0','08eea369','092763ef']

def find_info(uuid):
    info = {'uuid': uuid, 'in_main': False, 'in_dup': False, 'hash_row': None, 'file_path': None, 'file_exists': False, 'physical_location': None}
    if main_df is not None and 'uuid' in main_df.columns:
        row = main_df[main_df['uuid'].astype(str).str.strip() == uuid]
        if not row.empty:
            info['in_main'] = True
            info['main_row'] = row.iloc[0].to_dict()
    if dup_df is not None and 'uuid' in dup_df.columns:
        row = dup_df[dup_df['uuid'].astype(str).str.strip() == uuid]
        if not row.empty:
            info['in_dup'] = True
            info['dup_row'] = row.iloc[0].to_dict()
    if hashes_df is not None and 'uuid' in hashes_df.columns:
        row = hashes_df[hashes_df['uuid'].astype(str).str.strip() == uuid]
        if not row.empty:
            info['hash_row'] = row.iloc[0].to_dict()
            fp = Path(info['hash_row'].get('file_path', '')).expanduser()
            info['file_path'] = str(fp)
            if fp.exists():
                info['file_exists'] = True
                # classify where on disk
                if 'Duplicados' in [p.name for p in fp.parents]:
                    info['physical_location'] = 'Duplicados'
                else:
                    info['physical_location'] = 'other'
    return info

for u in sorted(set(main_only + dup_only)):
    info = find_info(u)
    print(u, info)