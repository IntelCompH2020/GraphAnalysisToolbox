#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main program for the RDIgraph analyzer

Created on June 18 2018

@author: Jesús Cid Sueiro

# For profiling: kernprof -lv mainRDIgraph.py ...

"""

import os
import argparse

# Local imports
from rdigraphs.menu_navigator.menu_navigator import MenuNavigator
from rdigraphs.labtaskmanager import LabTaskManager

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
project_path = args.p
if args.p is None:
    while project_path is None or project_path == "":
        project_path = input('-- Write the path to the project to load or '
                             'create: ')
if os.path.isdir(args.p):
    option = 'load'
else:
    option = 'create'
active_options = None
query_needed = False

# Create SuperGraph Project
# paths2data stores the path to all source data that might be needed by the
# task manager
paths2data = {'topicmodels': os.path.join(args.source, 'topic_models'),
              'agents': os.path.join(args.source, 'agents'),
              'ACL_models': os.path.join(args.source, 'ACL_models')}
tm = LabTaskManager(project_path, paths2data)

# ########################
# Prepare user interaction
# ########################

# paths2data stores the path to all folders that might be needed by the
# menu navigator. They may differ from those needed by the task manager
paths2data = {'graphs': os.path.join(project_path, 'graphs'),
              'bigraphs': os.path.join(project_path, 'bigraphs'),
              'topicmodels': os.path.join(args.source, 'topic_models'),
              'agents': os.path.join(args.source, 'agents'),
              'ACL_models': os.path.join(args.source,
                                         'ACL_models/models/tm'),
              'validation': os.path.join(project_path, 'output', 'validation')}
path2menu = 'lab_menu.yaml'

# ##############
# Call navigator
# ##############

menu = MenuNavigator(tm, path2menu, paths2data)
menu.front_page(title="RDI Graph Analyzer")
menu.navigate(option, active_options)
