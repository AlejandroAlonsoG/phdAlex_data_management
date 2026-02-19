from pathlib import Path
from data_ordering.pattern_extractor import extract_patterns

paths = [
	Path(r"D:\Dataset MUPA\Imágenes_MCCM_2009\Fotos_Abel_MCCM_09_11_06_\100OLYMP\PB090043.JPG"),
	# Camera-like filename where numeric-only inference shouldn't create MCCM-2009
	Path(r"D:\Dataset MUPA\Imágenes_MCCM_2009\some_folder\IMG_0001.JPG"),
	# Filename is just '2009' - should NOT produce MCCM-2009 because
	# the numeric looks like a year and the prefix is found in the path.
	Path(r"D:\Dataset MUPA\Imágenes_MCCM_2009\some_folder\2009.JPG"),
]

for p in paths:
	print('\nPATH =>', p)
	res = extract_patterns(p)
	print('  specimen_id:', res.specimen_id)
	print('  dates:', [(d.year,d.month,d.day,d.source.value,d.raw_match) for d in res.dates])
	print('  campaign:', res.campaign)
	print('  taxonomy_hints:', [(t.level,t.value,t.path_component) for t in res.taxonomy_hints])
	print('  numeric_ids:', [(n.numeric_id,n.is_likely_camera_number,n.raw_match) for n in res.numeric_ids])
