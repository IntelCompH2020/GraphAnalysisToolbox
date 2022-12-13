#!/bin/bash

source sphinx_settings/docs.env

pushd $DOCSDIR

make html

popd

echo
echo ========================
echo
echo trying to open $OUTPUT_INDEX...
echo

# derived from https://superuser.com/a/38989
case "$OSTYPE" in
   cygwin*)
      cmd /c start $OUTPUT_INDEX
      ;;
   linux*)
      xdg-open $OUTPUT_INDEX
      ;;
   darwin*)
      open $OUTPUT_INDEX
      ;;
esac
