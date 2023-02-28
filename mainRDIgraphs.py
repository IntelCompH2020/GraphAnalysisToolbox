#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main program for the RDIgraph analyzer

Created on June 18 2018

@author: Jesús Cid Sueiro

# For profiling: kernprof -lv mainRDIgraph.py ...

"""

import pathlib
import argparse

# Local imports
from rdigraphs.menu_navigator.menu_navigator import MenuNavigator
from rdigraphs.sgtaskmanager import SgTaskManager

# ########################
# Main body of application
# ########################

# ####################
# Read input arguments

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--p', type=str, default=None,
                    help="path to a new or an existing evaluation project")
parser.add_argument('--source', type=str, default='../source_data',
                    help="path to the source data folder")
args = parser.parse_args()

# Read project_path
path2project = args.p
if args.p is None:
    while path2project is None or path2project == "":
        project_path = input('-- Write the path to the project to load or '
                             'create: ')
path2project = pathlib.Path(path2project)

if path2project.is_dir():
    option = 'load'
else:
    option = 'create'
active_options = None
query_needed = False

# Create SuperGraph Project
path2source = pathlib.Path(args.source)
paths2data = {'topicmodels': path2source / 'topic_models',
              'agents': path2source / 'agents',
              'ACL_models': path2source / 'ACL_models'}
tm = SgTaskManager(path2project, paths2data, path2source)

# ########################
# Prepare user interaction
# ########################

paths2data = {'graphs': path2project / 'graphs',
              'bigraphs': path2project / 'bigraphs',
              'topicmodels': path2source / 'topic_models',
              'agents': path2source / 'agents'}
path2menu = pathlib.Path('config/options_menu.yaml')

# ##############
# Call navigator
# ##############

menu = MenuNavigator(tm, path2menu, paths2data)
menu.front_page(title="RDI Graph Analyzer")
menu.navigate(option, active_options)
