#!/usr/bin/env python
import yaml, os

config_data = yaml.load(
    open(
        "{0}/{1}".format(
            os.path.dirname(__file__), 
            "chime.yaml"
        ), 
        "r"
    )
)
