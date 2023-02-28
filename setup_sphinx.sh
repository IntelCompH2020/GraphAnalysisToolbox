#!/bin/bash

DOCSDIR=docs_all
PROJECT_NAME=SuperGraph
AUTHORS="J. Cid, M.A. Vázquez"
RELEASE_YEAR=2023
OUTPUT_INDEX=$DOCSDIR/build/html/index.html

mkdir $DOCSDIR

ROOT_DIR=$PWD

pushd $DOCSDIR

(echo y ; echo "$PROJECT_NAME" ; echo "$AUTHORS" ; echo $RELEASE_YEAR ; echo en)  | sphinx-quickstart

popd

cp sphinx_settings/* $DOCSDIR/source

# ln -sf "$ROOT_DIR"/sphinx_settings/*.rst "$ROOT_DIR"/$DOCSDIR/source

# pushd $DOCSDIR/source

# patch -p0 < "$ROOT_DIR"/sphinx_settings/conf.patch

# popd
