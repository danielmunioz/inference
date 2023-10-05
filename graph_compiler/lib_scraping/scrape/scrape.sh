#!/bin/bash

node=$1
to_scrape=$2
if [[ -n "$to_scrape" ]]
then
    echo "Attempting to scrape ${to_scrape}"
fi

apt-get install procps -y > /dev/null 2>&1

# install the local ivy_repo
cd /ivy/graph-compiler/ivy_repo
pip install -q --user -e .

# get chromium for selenium -> gpt token
bash /ivy/graph-compiler/lib_scraping/scrape/get_chrome.sh > /dev/null 2>&1

# install core requirements
cd /ivy/graph-compiler/lib_scraping/requirements
pip install -q -r requirements.txt

# create library-specific requirement files
while IFS= read -r line || [ -n "$line" ]
do
    # check if the line contains a package name
    if [[ $line == \#* ]]
    then
        # extract the package name from the line
        package=$(echo $line | cut -d ' ' -f 2)

        # create a new file for the package
        filename="requirements_${package}.txt"
        touch $filename

        # write the package header to the new file
        echo $line > $filename
    else
        # write the package requirement to the new file
        echo $line >> $filename
    fi
done < libraries_requirements.txt

cd /ivy/graph-compiler/
# if this is the starting node and we have scraped before, try to reuse them
if [[ $node == "1" && -d "lib_scraping/scrape/result" ]]
then
    echo "Starting node detected"
    rm -f lib_scraping/scrape/result/all_done
    for folder in lib_scraping/scrape/result/*
    do
        library=$(basename "$folder")

        # only scrape what we need if specified
        if [[ -n "$to_scrape" && "$to_scrape" == "$library" ]] || [[ -z "$to_scrape" ]]
        then
          rm -f "${folder}/requirements.txt"
        fi
    done
fi

# https://unix.stackexchange.com/questions/146756/forward-sigterm-to-child-in-bash/444676#444676
prep_term()
{
    unset term_child_pid
    unset term_kill_needed
    trap 'handle_term' TERM INT
}

handle_term()
{
    if [ "${term_child_pid}" ]; then
        kill -TERM "${term_child_pid}" 2>/dev/null
        sleep 60
        exit
    else
        term_kill_needed="yes"
    fi
}

wait_term()
{
    term_child_pid=$!
    if [ "${term_kill_needed}" ]; then
        kill -TERM "${term_child_pid}" 2>/dev/null
        sleep 60
        exit
    fi
    wait ${term_child_pid} || exit
    trap - TERM INT
    wait ${term_child_pid} || exit
}

# loop through the libraries
for file in lib_scraping/requirements/requirements_*.txt
do
    if [ -f "$file" ]
    then
        # install requirements
        pip install -q -r $file

        # extract basename
        name=$(basename "$file")

        # extract library name (remove prefix `requirements` and suffix `.txt`)
        library=${name:13:-4}

        if [[ -n "$to_scrape" && "$to_scrape" == "$library" ]] || [[ -z "$to_scrape" ]]
        then
           # if a library is scraped, it will have a requirements.txt inside its folder
           if [[ ! -e "lib_scraping/scrape/result/${library}/requirements.txt" ]]
           then
              prep_term
              python3 lib_scraping/scrape/scrape.py -n 0 "$library" &
              wait_term
           fi
        fi
    fi
done

# if everything is done, skip subsequent jobs
echo "Creating all_done"
touch lib_scraping/scrape/result/all_done
