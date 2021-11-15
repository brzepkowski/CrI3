#!/bin/bash

parameters_file=$1 # File containing all possible combinations of parameters J, L and Sz
finished_params_filename=$2

FILE=$parameters_file

while true
do
  number_of_jobs=$(qstat -u brzepkow | wc -l)
  if [ "$number_of_jobs" -lt "450" ]; then
    if [ -s $FILE ]; then
      line=$(head -1 $FILE)
      IFS=' '
      read -ra ADDR <<< "$line" # str is read into an array as tokens separated by IFS
      Lx="${ADDR[0]}"
      Ly="${ADDR[1]}"
      D="${ADDR[2]}"
      J="${ADDR[3]}"
      L="${ADDR[4]}"
      Sz="${ADDR[5]}"
      echo "D = $D, J = $J, L = $L, Sz = $Sz (Lx = $Lx, Ly = $Ly)"
      ./run_one_parameter_set.sh $Lx $Ly $D $J $L $Sz $finished_params_filename
      tail -n +2 "$FILE" > "$FILE.tmp" && mv "$FILE.tmp" "$FILE"
      sleep 0.5
    else
      echo "Empty file"
      break
    fi
  else
    echo "Too many jobs in the queue. Waiting..."
    sleep 60
  fi

done
