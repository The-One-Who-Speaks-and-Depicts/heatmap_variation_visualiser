import logging
import os

logger = logging.getLogger('heatmap_creator')

def load_variation_types(variation_file_path: str) -> list[str]:
    if not os.path.exists(variation_file_path):
        logger.warning("%s does not exists, no desired variation types detected", variation_file_path)
        return []
    with open(variation_file_path, 'r', encoding="utf-8") as inp:
        return [line.strip() for line in inp.readlines() if line and line.strip()]

