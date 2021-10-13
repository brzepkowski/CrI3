#!/bin/bash

CATALOG_NAMES=parameters_*

for catalog_name in $CATALOG_NAMES
do
  cd "$catalog_name"
  IFS='_' read -r -a array <<< "$catalog_name"
  D="${array[1]}"
  J="${array[2]}"
  L="${array[3]}"
  echo -e "${D}\t${J}\t${L}"
  # Save first line of the desired output file
  echo -e "${D}\t${J}\t${L}" | cat > "data_${D}_${J}_${L}"
  PARTIAL_DATA_NAMES=partial_data_*
  for partial_data_name in $PARTIAL_DATA_NAMES
  do
    cat "$partial_data_name" >> "data_${D}_${J}_${L}"
  done
  mv "data_${D}_${J}_${L}" ../
  cd ..
done
