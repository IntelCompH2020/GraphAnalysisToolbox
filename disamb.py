#! /usr/bin/env python3

import pathlib

import yaml
import pandas as pd

from rdigraphs.datamanager import callbacks

parameters_file = (pathlib.Path.cwd() / 'rdigraphs' / 'datamanager'
                   / 'parameters.yaml')

with open(parameters_file) as f:
    parameters = yaml.load(f)

path2dm = pathlib.Path.cwd() / 'rdigraphs' / 'datamanager'
projects_file = path2dm / 'tmp_projects' / 'projects.csv.gz'
researcher_project_file = (path2dm / 'tmp_projects'
                           / 'researcher_project.csv.gz')
organizations_file = path2dm / 'tmp_projects' / 'organization.csv.gz'

patstats_person_file = path2dm / 'tmp_patents' / 'person906.csv.gz'
patstats_person_application_file = (
    path2dm / 'tmp_patents' / 'person_application.csv.gz')

scopus_authorship_file = path2dm / 'tmp_publications' / 'authorship.csv.gz'

# ----------

projects = pd.read_csv(projects_file)
# researcher_project = pd.read_csv(researcher_project_file)
organizations = pd.read_csv(organizations_file)

patstats_person_application = pd.read_csv(
    patstats_person_application_file, dtype=str)

patstats_person = pd.read_csv(patstats_person_file, dtype=str)

scopus_authorship = pd.read_csv(scopus_authorship_file)

# ----------

# to simulate the original file
# projects = projects.drop(['disambiguated_id'], axis=1)
patstats_person = patstats_person.drop(['disambiguated_id'], axis=1)
patstats_person_application = patstats_person_application.drop(
    ['disambiguated_id'], axis=1)

parameters['disambiguation']['authors']['disambiguation_map'] = (
    ['rdigraphs', 'datamanager']
    + parameters['disambiguation']['authors']['disambiguation_map'])

parameters['disambiguation']['organizations']['disambiguation_map'] = (
    ['rdigraphs', 'datamanager']
    + parameters['disambiguation']['organizations']['disambiguation_map'])

callbacks.initialize(parameters['disambiguation'])

auth_disamb = callbacks.authors_disambiguator
org_disamb = callbacks.organizations_disambiguator

# res = callbacks.organizations_disambiguator.projects_organizations(
#     organizations)
# res = callbacks.organizations_disambiguator.projects_projects(projects)

# res = callbacks.authors_disambiguator.patstats_person(patstats_person)
# res2 = callbacks.organizations_disambiguator.patstats_person(patstats_person)

res = callbacks.organizations_disambiguator.scopus_authorship(
    scopus_authorship)
