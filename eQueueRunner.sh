#!/bin/bash
# Reads all files matching experimentQueue/*.t and executes
# their contents. The file is then deleted and output is logged
# in .log file with same name.
# Script keeps restarting every 10 seconds to look for new tasks.

if [ -e experimentQueue/*.t ]
then
  for k in experimentQueue/*.t; do
    echo "Executing" $k
    chmod a+x $k
    cat $k >> $k.log
    $k >> $k.log
    rm -f $k
  done
fi

echo "Tasks complete, waiting for new experiments..."
sleep 10
script=$(readlink -f "$0")
exec "$script"

