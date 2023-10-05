# Scrape guide

## Structure

```bash
lib_scraping/scrape/
├── get_chrome.sh        # helper to get chromium on linux
├── get_poe_tokens.py    # helper to get gpt tokens
├── README.md
├── scrape.py            # local starting point
└── scrape.sh            # actions starting point
```

## Usage

1. Using Github Actions:
    - Launch directly using [workflows/scrape.yml](https://github.com/unifyai/graph-compiler/actions/workflows/scrape.yml) on the web interface.
    - Inputs:
        - Python name of a specific library to scrape (default: `None`, continue scraping for every library).
        - Run ID of the scrape artifact to download and continue scraping from (default: `None`, continue from the latest scrape).
    - Outputs: `scrape-artifacts.zip`
    - Schedule: at 00:00 UTC on the 1st day of every 3rd month.

2. Run locally:
    - Launch directly using `python lib_scraping/scrape/scrape.py`.
    - Inputs:
        - `--debug`: Print our GPT queries (default: `None`, only prints GPT's answers).
        - `-n`, `--num-processes`: Number of parallel processes, set as 0 to use the maximum amount of `num_logical-2` (default: `1`, no multiprocessing).
        - `library`: Python name of the library to scrape for.
    - Outputs: `lib_scraping/scrape/result/*`

## Output format
```python
# library.dil
{
    'frameworkA':
        {
            'function':
                {
                    'path.to.fnB':
                        {
                            'code': 'fnB(*args, **kwargs)',
                            'imports': ['path.to', 'framework', ...]
                        },
                    ...
                },
            'class':
                {
                    'path.to.clsC':
                        {
                            'code': 'clsC.init_method(*init_args, **init_kwargs).call_method(*call_args, **call_kwargs)',
                            'imports': ['path.to', 'framework', ...]
                        },
                    ...
                }
        },
    ...
}

# library_non_tensor.dil
Similar to library.dil but without the framework layer.
```

## Notes

1. **Adding a library**: set the requirements by appending to `lib_scraping/requirements/libraries_requirements.txt` in the correct format of:
```bash
# library_name - [frameworks_included] - github:some_org/library
dependency1
dependency2
-e git+https://github.com/some_org/dependency3.git#egg=dependency3
--find-links https://someweb.com/dependency4/...
dependency4
...
```
As we are using `pip freeze -r libraries_requirements.txt` to freeze the environment according to a template, setting a dependency in an unconventional way (but still installable) would not capture the correct version / uninstallable later on.

Examples:
- **Wrong**: `git+https://github.com/facebookresearch/detectron2.git` -> `detectron==0.6`
- **Correct**: `-e git+https://github.com/facebookresearch/detectron2.git#egg=detectron2` -> `detectron2 @ git+https://github.com/facebookresearch/detectron2.git@2c6c38` [Why?](https://stackoverflow.com/questions/34881247/pip-egg-name-for-editable-dependency-changes)
- **Wrong**: `numpy_ml[rl]` -> Nothing
- **Correct**: `gym \n matplotlib` -> `gym==0.26.2 \n matplotlib==3.7.1` [Why?](https://github.com/ddbourgin/numpy-ml/blob/b0359af5285fbf9699d64fd5ec059493228af03e/setup.py#LL33C50-L33C50)

2. **Adding a framework**: import the framework, set the module type (if exists) and tensor type of the framework in `scrape.py`:
```python
# -------- ADD A FRAMEWORK TO SCRAPE FOR HERE --------
...
import fw

module_types = (..., fw.model)
tensor_types = {
   ...,
   "fw": fw.tensor,
}
# ----------------------------------------------------
```