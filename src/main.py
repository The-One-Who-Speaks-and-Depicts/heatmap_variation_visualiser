import argparse
import logging
import os

from pipeline import run_pipeline

logger = logging.getLogger('heatmap_creator')
logger.setLevel(logging.DEBUG)
log_handler = logging.StreamHandler()
log_formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s: %(message)s",
                              datefmt='%Y-%m-%d %H:%M:%S')
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)

def main(args):
    run_pipeline(args)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", "-t", help="Type of data", choices=[
        "ua_gec",
        "ua_ner",
        "automatic_thematic_modelling",
        "text_variation",
        "pos",
        "lemma"], default="ua_gec", required=True)
    parser.add_argument("--rapidity_rate", "-r", help="Rapidity rate", default="3")
    parser.add_argument("--clustered", "-c", choices=["1", "0"], help="Clustered", default="0")
    parser.add_argument("--source_directory", "-s", help="Source directory")
    parser.add_argument("--variation_type", "-v", help="File with types of variation to visualise, one by line")
    parser.add_argument("--text_separator", "-ts", help="Text separator for automatic thematic modelling", default="")
    parser.add_argument("--lda_num_topics", "-ldant", help="Number of topics for automatic thematic modelling", default="10")
    parser.add_argument("--lda_alpha", "-ldaa", help="Alpha value for automatic thematic modelling", default="auto")
    parser.add_argument("--lda_passes", "-ldap", help="Number of passes for automatic thematic modelling", default="500")
    parser.add_argument("--lda_epochs", "-ldae", help="Number of epochs for automatic thematic modelling", default="300")
    parser.add_argument("--lda_required_num", "-ldarn", help="Number of topics to gather for automatic thematic modelling", default="10")
    parser.add_argument("--lda_required_start", "-ldars", help="The first topic to gather for automatic thematic modelling", default="0")
    parser.add_argument("--output_directory", "-out", help="Directory to store the results of heatmap visualisation", default=os.getcwd())
    parser.add_argument("--max_rows", "-mr", help="Maximum rows in heatmap visualisation plot", default="1")
    parser.add_argument("--is_heat_continuous", "-cont", help="Controls, whether heatmap visualisation is continious, or split in rows", choices=["True", "False"], default="False")
    parser.add_argument("--func_path", "-fp", help="Full path to the function that equalizes the rows in the resulting matrix", default="numpy.max")
    parser.add_argument("--percentile", "-pt", help="If func_path equals to numpy.percentile, sets percentile")
    args = parser.parse_args()
    main(args)
