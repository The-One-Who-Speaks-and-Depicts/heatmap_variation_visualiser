import logging
import os

from dataclasses import dataclass

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

logger = logging.getLogger("heatmap_creator")

@dataclass
class HeatmapParams():
    output_dir: str = ""
    max_rows: int = 1
    is_continuous: bool = False

def chunk_matrix(matrix: list[list], max_rows: int) -> list[np.ndarray[np.ndarray]]:
    matrix = np.asarray(matrix)
    if not all(
        isinstance(i, np.ndarray) for i in matrix
    ):
        raise ValueError("matrix is not list of lists")
    if len(matrix) < 1:
        raise ValueError("empty matrix")
    if not all(
        all(
            isinstance(j, np.signedinteger) for j in i
        ) and len(i) > 0 for i in matrix
    ):
        raise ValueError("Every row of matrix should be a non-empty array of integers")
    if not all(
        len(i) == len(matrix[0]) for i in matrix
    ):
        raise ValueError("Matrix rows should have equal length")
    if not isinstance(max_rows, int):
        raise ValueError("max_rows should be an instance of integer")
    if max_rows < 1:
        raise ValueError("max_rows should be positive")
    if max_rows > len(matrix):
        logger.warning('Provided number of rows %s to visualise is higher than the length of matrix % s; setting number of rows to length of matrix', max_rows, len(matrix))
        max_rows = len(matrix)
    chunked_matrix = np.array_split(matrix, len(matrix) // max_rows)
    logger.debug(
        "Matrix split into %s chunks of %s x %s size", 
        len(chunked_matrix), 
        len(chunked_matrix[0]), len(chunked_matrix[0][0])
        )
    return chunked_matrix

def create_heatmap_from_matrix(
        matrix: np.ndarray[np.ndarray],
        output_dir: str, is_continuous: bool, 
        idx: int, dataset_part_idx: int = 0
        ) -> None:
    logger.debug("Storing vis %s in %s", 
                idx, os.path.join(output_dir, "chunk_" + str(dataset_part_idx) + "_" + str(idx)))
    plt.figure(dpi=300, figsize=[12.8, 9.6])
    if is_continuous:
        plt.imshow(matrix, cmap='viridis')
        plt.colorbar(shrink=0.5)
    else:
        sns.heatmap(matrix, linewidth=0.5)
    plt.savefig(os.path.join(output_dir, "chunk_" + str(dataset_part_idx) + "_" + str(idx)))
    plt.close()

def create_heatmap(matrix: list[list], params = HeatmapParams(), dataset_part_idx: int = 0) -> None:
    if not os.path.exists(params.output_dir):
        raise ValueError("Directory for results storage does not exist")
    chunks = chunk_matrix(matrix, params.max_rows)
    for idx, val in enumerate(chunks):
        create_heatmap_from_matrix(
            val, 
            params.output_dir, params.is_continuous,
            idx, dataset_part_idx)
        
def create_heatmap_with_labels(matrix: list[list], symbols: list[list], params = HeatmapParams(), dataset_part_idx: int = 0) -> None:
    logger.debug("Storing vis %s in %s", 
                dataset_part_idx, os.path.join(params.output_dir, "chunk_" + str(dataset_part_idx) + "_" + str(dataset_part_idx)))
    
    fig = go.Figure(data=go.Heatmap(
                    z=matrix,
                    text=symbols,
                    texttemplate="%{text}",
                    textfont={"size":15}), layout=go.Layout(
                        title = "title",
                        yaxis=dict(visible=False,autorange='reversed')
                    ))
    fig.update_layout(
    autosize=False,
    width=4000,
    height=2000,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
    paper_bgcolor="LightSteelBlue",
)
    fig.write_image(os.path.join(params.output_dir, "chunk_" + str(dataset_part_idx) + ".png"), format="png")