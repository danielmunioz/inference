# Compile guide

## Structure

```bash
lib_scraping/compile/
├── compile.py           # local starting point
├── compile.sh           # actions starting point
├── README.md
└── upload.py            # upload results to gc repo issues, frequency dashboard
```

## Usage

1. Using Github Actions:
    - Launch directly using [workflows/compile.yml](https://github.com/unifyai/graph-compiler/actions/workflows/compile.yml) on the web interface.
    - Inputs:
        - Python name of a specific library to compile (default: `None`, compile every scraped library).
        - Run ID of the scrape artifact to download and compile from (default: `None`, compile from the latest scrape).
    - Outputs: `compile-artifacts.zip`
    - Schedule: at 00:00 UTC on every Saturday.

2. Run locally:
    - Launch directly using `python lib_scraping/compile/compile.py`.
    - Inputs:
        - `-d`, `--download`: Download latest scrape artifacts and install environment of the specified library (default: `None`, no download).
        - `-b`, `--benchmark`: Benchmark time performance of native, native compiled, ivy compiled, transpiled and transpiled compiled.
        - `-n`, `--num-processes`: Number of parallel processes, set as 0 to use the maximum amount of `num_physical-1` (default: `1`, no multiprocessing).
        - `library`: Python name of the library to compile.
    - Outputs: `lib_scraping/compile/result/*`

## Output format
```
# library.csv
Functions, Compile Frequencies, Source Frequencies, Missing Functions, Compile Frequencies, Source Frequencies
fnA, 5, 5, fnB, 1, 1
fnB, 1, 1, , , 
...

# library_failed.txt
`requirements.txt`
library==version
...

FrameworkA
n/n Functions failed to compile
n/n Classes failed to compile
n/n Function compile outputs unmatched
n/n Class compile outputs unmatched
n/n Functions compiled in FrameworkA but can't be transpiled to
n/n Classes compiled in FrameworkA but can't be transpiled to
n/n Function transpile outputs unmatched between FrameworkA and
n/n Class transpile outputs unmatched between FrameworkA and

----- Not related to compiling
n went Out-Of-Memory
n had an Import issue
n only had a Tensor in Attribute
```