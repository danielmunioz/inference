#!/bin/bash

to_compile=$1
if [[ -n "$to_compile" ]]
then
    echo "Attempting to compile ${to_compile}"
fi

# https://github.com/actions/runner-images/issues/6775
git config --global --add safe.directory /ivy/graph-compiler

# install the local ivy_repo
cd /ivy/graph-compiler/ivy_repo
pip install -q --user -e .

# install core requirements
cd /ivy/graph-compiler/lib_scraping/requirements
pip install -q -r requirements.txt

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

cd /ivy/graph-compiler/
# loop through the library folders
for folder in lib_scraping/scrape/result/*
do
    if [[ -d "${folder}" && -e "${folder}/requirements.txt" ]]
    then
        # extract library name
        library=$(basename "$folder")

        if [[ -n "$to_compile" && "$to_compile" == "$library" ]] || [[ -z "$to_compile" ]]
        then
            # install requirements
            pip install -q -r "${folder}/requirements.txt"

            # if a library is compiled, it will have a result file inside its folder
            if [[ ! -e "lib_scraping/compile/result/${library}/frequency.csv" ]]
            then
                # continue compiling from the previous commit of the compiler repo,
                # in case the jobs spanned multiple commits
                if [[ -e "lib_scraping/compile/result/${library}/start_commit.txt" ]]
                then
                    git checkout -f "$(cat "lib_scraping/compile/result/${library}/start_commit.txt")"
                fi

                prep_term
                python3 lib_scraping/compile/compile.py -b -n 0 "$library" &
                wait_term
            fi

            # if a library is uploaded after compiling, it will have a temp file inside its folder
            if [[ -e "lib_scraping/compile/result/${library}/frequency.csv" && ! -e "lib_scraping/compile/result/${library}/uploaded" ]]
            then
                prep_term
                python3 lib_scraping/compile/upload.py $library &
                wait_term
            fi
        fi
    fi
done

# if everything is done, skip subsequent jobs
echo "Creating all_done"
touch lib_scraping/compile/result/all_done
