"""
Run the uncoverml pipeline for super-learning and prediction.

.. program-output:: uncoverml --help
"""

import logging
import pickle
from pathlib import Path
import os
import re
import warnings

import click
import yaml

import uncoverml.config
import uncoverml.scripts

_logger = logging.getLogger(__name__)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


# print(dir(uncoverml.scripts.cli))

def main(yaml_file, partitions):
    """
    """

    def _grp(d, k, msg=None):
        """
        Get required parameter.
        """
        try:
            return d[k]
        except KeyError:
            if msg is None:
                msg = f"Required parameter {k} not present in config."
            _logger.exception(msg)
            raise

    uncoverml.config.Config._configure_pyyaml()
    with open(yaml_file, 'r') as f:
        try:
            s = yaml.safe_load(f)
        except UnicodeDecodeError:
            if yaml_file.endswith('.model'):
                _logger.error("You're attempting to run uncoverml but have provided the "
                              "'.model' file instead of the '.yaml' config file. The predict "
                              "now requires the configuration file and not the model. Please "
                              "try rerunning the command with the configuration file.")
            else:
                _logger.error("Couldn't parse the yaml file. Ensure you've provided the correct "
                              "file as config file and that the YAML is valid.")
    learn_lst = _grp(s, 'learning', "'learning' block must be provided when superlearning.")
    # with open("./File.yaml", 'w') as f:
        # yaml.dump(s['learning'], f, default_flow_style=False, sort_keys=False)
    s.pop("learning")
    ddd = {}
    for alg in learn_lst:
        ddd.update({"learning": alg})
        ddd.update(s)
        ddd["output"]["directory"] = f"./{alg['algorithm']}_out"
        ddd["output"]["model"] = f"./{alg['algorithm']}_out/{alg['algorithm']}.model"
        ddd["pickling"]["covariates"] = f"./{alg['algorithm']}_out/features.pk"
        ddd["pickling"]["targets"] = f"./{alg['algorithm']}_out/targets.pk"
        with open(f"{alg['algorithm']}.yml", 'w') as yout:
            yaml.dump(ddd, yout, default_flow_style=False, sort_keys=False)
        uncoverml.scripts.learn.callback(f"{alg['algorithm']}.yml", partitions)

        # predicting
        # pb = _grp(s, 'prediction', "'prediction' block must be provided.")
        # self.geotif_options = pb.get('geotif', {})
        # self.quantiles = _grp(pb, 'quantiles', "'quantiles' must be provided as part of prediction block.")
        # self.outbands = _grp(pb, 'outbands', "'outbands' must be provided as part of prediction block.")
        # self.thumbnails = pb.get('thumbnails', 10)
        # self.bootstrap_predictions = pb.get('bootstrap')

        retain = None

        mb = s.get('mask')
        if mb:
            mask = mb.get('file')
            if not os.path.exists(mask):
                raise FileNotFoundError("Mask file provided in config does not exist. Check that the 'file' property of the 'mask' block is correct.")
            retain = _grp(mb, 'retain', "'retain' must be provided if providing a prediction mask.")
        else:
            mask = None

        # if self.krige:
            # Todo: don't know if lon/lat is compulsory or not for kriging
            # self.lon_lat = s.get('lon_lat')
        # else:
            # self.lon_lat = None

        try:
            uncoverml.scripts.predict.callback(f"{alg['algorithm']}.yml", partitions, mask, retain)
        except TypeError:
            _logger.error(f"Learner {alg} cannot predict")
    # uncoverml.scripts.cli.commands['learn']("./aaa.yml", partitions)
    # uncoverml.scripts.learn.callback("./aaa.yaml", partitions)
    return
