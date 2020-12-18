#!/usr/bin/env python
import yaml, os

general = yaml.load(
    open(
        "{0}/{1}".format(
            os.path.dirname(__file__), 
            "general.yaml"
        ), 
        "r"
    ),
    Loader=yaml.FullLoader
)

telescopes = yaml.load(
    open(
        "{0}/{1}".format(
            os.path.dirname(__file__), 
            "telescopes.yaml"
        ), 
        "r"
    ),
    Loader=yaml.FullLoader
)
