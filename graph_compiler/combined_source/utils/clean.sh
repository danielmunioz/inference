#!/bin/bash

for file in *.so
do
    # Get the filename without the .so extension
    filename=$(basename "$file")
    Shared_name=$(basename "$file" .cpython-310-x86_64-linux-gnu.so)
    echo $filename
    echo $Shared_name
    
        # Check if the new filename ends with cpython-38-x86_64-linux-gnu.so
    if [[ "$filename" == *cpython-310-x86_64-linux-gnu.so ]]; then
        # Rename the file to remove the cpython-38-x86_64-linux-gnu.so suffix
        mv "$filename" "$Shared_name.so" 
    fi
done
