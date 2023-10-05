import gspread
from gspread_formatting import (
    cellFormat,
    color,
    format_cell_range,
    batch_updater,
    set_column_width,
    set_frozen,
)
import pandas as pd
import github
import argparse
import csv
import re
import sys
import random
import os
from typing import Tuple

sys.path.append("lib_scraping/scrape/")
from scrape import frameworks

# gcloud - spreadsheet - function dashboard access
service_account = {
    "type": "service_account",
    "project_id": "cloud-functions-360207",
    "private_key_id": "695dc43bbe764b21dc343d37af73d3daec91fbf0",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQDhxFFDQ4+l1Q6o\nxLiBl/ftOaifg11REmUTCb7bR4ZRC23jwXFXGKGCU/U1y7kk6mXEU6mHO7AIvl7r\nmhBxq0pzyA7sk1yHTlen1nC8YkMKVrSSVLR2eQZR/0nub7obyS/f99brHmsF84wU\nv34s73Azz8Cb9tNYhkO5YYdohl6PwWerhBvVFB65lk3/qq8IcTdBwO4ERpAM/dDl\nnXqvs8nJQEq043AoiksSysxlFBCi4gpNPLuGBg6267XzdavgAdRTAoy7dP9ORQSA\naLxJB9XiDGCSdTPODv2BsTHFO1cQzWkY2hdYQYN1zjWn9lx+327mssaUbIS8GeNb\n2m2LTWJdAgMBAAECggEABB2MRRKsjtKqR8sx0tKJzclhu8PZz5IrUgEsOVQqZfyB\nMw5+0ESuBjgDrdVqHIeb6P8ea/O1Iux3df0ZUVa7DqiMe+VrlLk9BOpcvgf4UWF6\nieCHgdGO2vx+Md2LBRnyG7ecmeCBONSAKFoQq0u53gcd0d0z7CnvmGfakrFaWwco\n5GD6Y0nnRNNbBdJ2VMWMMj3ijms+FdrbfA0UQHmXuqmsjNzFyUZBgmCl6CJW/Oft\nZsz0TllWORzFCyPQrOYIfCDDPgjaehPF29NdEfRk9ivSdviaPWIBjyuHpKnLvl7n\n4C/WovH1BWVPf8DaD4msspRVd93NfJeyDG7KIN9NiQKBgQD7lKvJU8hRITUermQu\n3RJB0Xn2RF6cr2lpuSCV2b/ZvazcNf5PvfL88HNeqSjXGksSZtzckK47nmuX8R/y\nTgG58CZ2dWKSawAANUoYPovvqcVqKcp5eCuhMDiC9x/VCrhXSRx7dHOvx0SUQqy2\nPOT+HgJkawGz/9hCFCQqoteQ6QKBgQDlu5B9OEzv7cS46AUe18rLsooP3fMXvdm3\ninuQjh0fgk1IC2N1+MX1l/aH2VSpKE55DnH6FixypQAZkj/SQOEsjjM73wr58oxm\nPAHwDlPNaZ7jYaft18pzhSZDufzy6FFWyufJYwoPWZC8OLyk/CUOFOpcGOxaK6MB\nM02yYOT9VQKBgFHoWHmmrmxDjIDdtD5RuXT2V2fnYIpQzdge7s364+xnRZU4ewDJ\nTggt3NHv7x0BMXnfRX8GF8PPUyZX2dfQr90yo8MjeqFC9vAaaXI1QugXdO+YhZRA\nnKvRAEUbYiDBabz5T62d/2A2V0yR3JtEfiWB7bN150sMPANffVroQ5ipAoGAWMcI\n6TsOkFGECiivgeHGXr1aGROeU3hsYD9FzPD+VCTYlJTCFN7UMTpObOURkGUhHir0\n5L4Y4xzcUwVvYGLuIXe6WNKyvTB8DS33WbtPqzu7yQb+DC2t8MJtrRJ8q6oXdMDo\nnayGQLRN+E68p81AzJZMktaWz6m5TkdzKRHErBkCgYAEtBt51xFDOpqq18OfE+xT\nu7wM3B6cpC/wLvoMQZiS5rXwU53sspu4+GHVdu2uqueLOKIn8WDyEuZtsBHMPLDp\nbVTXQb9rhZKbsDKfvwWtYm+CfpM6sYhf4/UcTo62/d9st5m5Ddj+SOrd4seE6+kl\nuCbx8VGw7metdO4aBIWJLw==\n-----END PRIVATE KEY-----\n",
    "client_email": "cloud-functions-360207@appspot.gserviceaccount.com",
    "client_id": "107816566524143828868",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/cloud-functions-360207%40appspot.gserviceaccount.com",
}
ivy_leaves_token = "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"
function_spreadsheet_name = "Library Function Frequencies and Missing Functions"
benchmark_spreadsheet_name = "Library Compile and Transpile Benchmarking"
gc_team = ["mattbarrett98", "hmahmood24", "Sam-Armstrong"]


def extract_data(function_spreadsheet: gspread.spreadsheet.Spreadsheet) -> pd.DataFrame:
    """Get list of all sheets except for "Main" and "Repo_Stats" """
    sheets = function_spreadsheet.worksheets()
    sheets = [sheet for sheet in sheets if sheet.title not in ["Main", "Repo_Stats"]]

    result_df = pd.DataFrame()
    for sheet in sheets:
        # Read the data from sheet
        sheet_name = sheet.title
        each_sheet = function_spreadsheet.worksheet(sheet_name)
        each_data = each_sheet.get_all_values()

        # Create a pandas DataFrame from the data
        lib_df = pd.DataFrame(each_data[1:], columns=each_data[0])

        # Create an empty dictionary to store the data
        data_dict = {
            "Framework": [],
            "Function Name": [],
            "Compile Frequency": [],
            "Source Frequency": [],
            "Missing": [],
            "Library": [],
        }

        # Set to store function names added from the "Missing" column
        added_from_missing = set()

        # Process the data from the DataFrame
        for _, row in lib_df.iterrows():
            # Add missing function data (if any)
            if row["Missing Functions"]:
                data_dict["Framework"].append(row[4])
                data_dict["Function Name"].append(row[5])
                data_dict["Compile Frequency"].append(int(row[6]))
                data_dict["Source Frequency"].append(int(row[7]))
                data_dict["Missing"].append(
                    True
                    if not any(
                        filter in row["Missing Functions"] for filter in ("._", "utils")
                    )
                    else False
                )
                data_dict["Library"].append(sheet_name)
                added_from_missing.add(row["Missing Functions"])

            # Add function data if it was not added from the "Missing" column
            if row["Functions"] not in added_from_missing:
                data_dict["Framework"].append(row[0])
                data_dict["Function Name"].append(row[1])
                data_dict["Compile Frequency"].append(int(row[2]))
                data_dict["Source Frequency"].append(int(row[3]))
                data_dict["Missing"].append(False)
                data_dict["Library"].append(sheet_name)

        # Convert the dictionary to a pandas DataFrame
        result_df = pd.concat([result_df, pd.DataFrame(data_dict)])
    return result_df


def extract_repo_stats(
    function_spreadsheet: gspread.spreadsheet.Spreadsheet,
    ivy_leaves: github.MainClass.Github,
) -> None:
    """
    Get repo stars and contributors count and included frameworks
    so we can filter for them in the dashboard.
    """
    lib_info = list()
    with open("lib_scraping/requirements/libraries_requirements.txt", "r") as f:
        for line in f:
            if line.startswith("#"):
                # format: # library - [fwA, fwB, ...] - github
                lib_info.append(
                    re.search("^# (.*) - \[(.*)\] - github:(.*)$", line).groups()
                )

    repo_stats = function_spreadsheet.worksheet("Repo_Stats")
    repo_stats.clear()
    # set headers
    headers = ["Library/Sheet_Name", "Owner/Repo", "Stars", "Contributors"]
    headers.extend(frameworks)
    repo_stats.append_row(headers)

    for repo_name, repo_fw, repo_git in lib_info:
        repo = ivy_leaves.get_repo(repo_git)
        # get the star count and the number of contributors
        star_count = repo.stargazers_count
        contributor_count = repo.get_contributors().totalCount
        # get included frameworks
        repo_fw = [fw in repo_fw for fw in frameworks]
        row = [repo_name, repo_git, star_count, contributor_count]
        row.extend(repo_fw)
        repo_stats.append_row(row, value_input_option="USER_ENTERED")
    print("Done extract_repo_stats")


def update_main_and_repo_sheets(
    function_spreadsheet: gspread.spreadsheet.Spreadsheet,
) -> pd.DataFrame:
    """
    Aggregate all library frequencies into one sheet "Main"
    to provide data for function usage dashboard.
    """
    result_df = extract_data(function_spreadsheet)

    # Get the 'Main' sheet
    main_sheet = function_spreadsheet.worksheet("Main")
    main_sheet.clear()
    main_sheet.insert_row(result_df.columns.tolist(), 1)
    data_to_upload = result_df.values.tolist()
    main_sheet.append_rows(data_to_upload)
    print("Done update_main_and_repo_sheets")
    return result_df


def update_missing_frontend_issue(
    result_df: pd.DataFrame, ivy_repo: github.Repository.Repository
) -> None:
    """
    Update the missing frontend based on priority in each framework.
    """
    frontend_todo_issue = ivy_repo.get_issue(15022)
    original_body = frontend_todo_issue.body
    original_body_by_fw = dict()
    current_fw = None
    for line in original_body.split("\n"):
        if line.startswith("#"):
            # format: # Framework
            current_fw = line.split()[1].lower()
            original_body_by_fw[current_fw] = list()
            continue

        if current_fw != None:
            original_body_by_fw[current_fw].append(line)

    new_body = (
        "Below is the list of missing frontend functions needed to compile "
        "popular libraries, **ranked from high to low priority**. "
        "This list should be updated once per month.\n\n"
    )
    for fw in frameworks:
        # filter by each framework
        missing_df = result_df[
            (result_df["Function Name"].str.startswith(fw))
            & (result_df["Missing"] == True)
        ]
        missing_df = (
            missing_df.groupby("Function Name")["Compile Frequency"].sum().reset_index()
        )
        missing_df["Percentage"] = (
            missing_df["Compile Frequency"] / missing_df["Compile Frequency"].sum()
        ) * 100
        missing_df = missing_df.sort_values("Percentage", ascending=False)

        new_body += f"# {fw.title()}\n"
        for _, row in missing_df.iterrows():
            fn = row["Function Name"].replace(r"_", r"\_")

            # parse the original body to keep track of implementing frontends
            # because this todolist is prioritized
            for original in original_body_by_fw[fw]:
                # keep a html comment after the issue line
                # so we don't have to parse and open an issue to get its name
                # because it's now replaced by # - [] #issue_number
                if f"<!--{fn}-->" in original and original.startswith("- [ ]"):
                    new_body += original + "\n"
                    break
            else:
                new_body += f"- [ ] {fn} <!--{fn}-->\n"

        # push completed frontends down
        for original in original_body_by_fw[fw]:
            if original.startswith("- [x]") and original not in new_body:
                new_body += original + "\n"

    frontend_todo_issue.edit(body=new_body)
    print("Done update_missing_frontend_issue")


def update_or_create_gc_issue(
    library: str, gc_repo: github.Repository.Repository
) -> None:
    """
    Update the failing functions and classes issue in graph-compiler repo.
    If it doesn't exist, create one and update it, and assign it to
    one of the graph-compiler team members.
    Then make ivy_leaves mention the assigned in a comment
    to remind them that the issue is updated.
    """

    def split_at_last_newline(string: str, char_limit: int) -> Tuple[str, str]:
        last_newline_index = string.rfind("\n", 0, char_limit)
        if last_newline_index == -1:
            split_index = char_limit
        else:
            split_index = last_newline_index
        substring1 = string[:split_index]
        substring2 = string[split_index + 1 :]
        return substring1, substring2

    with open(f"lib_scraping/compile/result/{library}/failed.txt", "r") as fp:
        issue_body = fp.read()

    # github body limit
    char_limit = 65000
    issue_body, issue_body_remaining = split_at_last_newline(issue_body, char_limit)

    scrape_issue = [
        x
        for x in gc_repo.get_issues()
        if x.title == f"{library.capitalize()} failed functions and classes"
    ]

    if len(scrape_issue) == 0:
        scrape_issue = gc_repo.create_issue(
            f"{library.capitalize()} failed functions and classes",
            body=issue_body,
            assignees=[
                "xoiga123",
                random.choice(gc_team),
            ],
        )
    else:
        scrape_issue = scrape_issue[0]
        scrape_issue.edit(body=issue_body)

    # delete previous Continued comments (github limit)
    for comment in scrape_issue.get_comments():
        if comment.user.login == "ivy-leaves" and comment.body.startswith("Continued"):
            comment.delete()

    # delete previous @mention comments
    for comment in scrape_issue.get_comments():
        if comment.user.login == "ivy-leaves" and "Please review" in comment.body:
            comment.delete()

    # new Continued comments
    while len(issue_body_remaining):
        issue_body_continued, issue_body_remaining = split_at_last_newline(
            issue_body_remaining, char_limit
        )
        if len(issue_body_continued):
            scrape_issue.create_comment("Continued\n---------\n" + issue_body_continued)

    # new @mention comment
    assignees = ["@" + assignee.login for assignee in scrape_issue.assignees]
    scrape_issue.create_comment(
        " ".join(assignees)
        + "! Please review the newest compile result. Thank you :hugs:"
    )

    print("Done update_or_create_gc_issue", library)


def get_or_create_library_sheet(
    library: str, spreadsheet: gspread.spreadsheet.Spreadsheet
) -> gspread.worksheet.Worksheet:
    """
    Returns the specified library sheet in the spreadsheet.
    If it doesn't exist, create one and return it.
    """
    try:
        library_sheet = spreadsheet.worksheet(library)
        library_sheet.clear()
    except Exception:
        # 1 row and 1 column
        # this will be expanded when we add new rows anyway
        library_sheet = spreadsheet.add_worksheet(library, "1", "1")

    return library_sheet


def upload_function_frequencies(
    library: str, function_spreadsheet: gspread.spreadsheet.Spreadsheet
) -> None:
    library_sheet = get_or_create_library_sheet(library, function_spreadsheet)

    function_spreadsheet.values_update(
        library_sheet.title,
        params={"valueInputOption": "USER_ENTERED"},
        body={
            "values": list(
                csv.reader(open(f"lib_scraping/compile/result/{library}/frequency.csv"))
            )
        },
    )
    print("Done upload_function_frequencies", library)


def upload_benchmark_result(
    library: str, benchmark_spreadsheet: gspread.spreadsheet.Spreadsheet
) -> None:
    library_sheet = get_or_create_library_sheet(library, benchmark_spreadsheet)

    benchmark_spreadsheet.values_update(
        library_sheet.title,
        params={"valueInputOption": "USER_ENTERED"},
        body={
            "values": list(
                csv.reader(open(f"lib_scraping/compile/result/{library}/benchmark.csv"))
            )
        },
    )
    format_benchmark_sheet(library_sheet)
    print("Done upload_benchmark_result", library)


def format_benchmark_sheet(library_sheet: gspread.worksheet.Worksheet) -> None:
    """Highlight the fastest framework in green."""
    set_column_width(library_sheet, "B", 400)
    set_frozen(library_sheet, rows=1, cols=2)

    # reset everything to white
    default_format = cellFormat(
        backgroundColor=color(1, 1, 1),
    )
    max_col = gspread.utils.rowcol_to_a1(1, library_sheet.col_count)[:-1]
    format_cell_range(library_sheet, f"A:{max_col}", default_format)

    # highlight fastest to green
    highlight_format = cellFormat(
        backgroundColor=color(51 / 255, 255 / 255, 51 / 255),
    )
    batch = batch_updater(library_sheet.spreadsheet)
    for row, row_values in enumerate(library_sheet.get_values(), start=1):
        if row == 1:
            # skip header
            continue

        fastest_col = 3
        fastest_time = 999999

        for col, time in enumerate(row_values[2::2], start=3):
            # skip Framework, Callable columns
            # and only compare Mean columns (skip Std columns)
            try:
                if len(time) and float(time) < fastest_time:
                    fastest_time = float(time)
                    fastest_col = 2 * col - 3
            except Exception as e:
                print(repr(e))

        mean_cell = gspread.utils.rowcol_to_a1(row, fastest_col)
        std_cell = gspread.utils.rowcol_to_a1(row, fastest_col + 1)
        batch.format_cell_range(
            library_sheet, f"{mean_cell}:{std_cell}", highlight_format
        )

    try:
        batch.execute()
    except Exception as e:
        # if sheet is empty
        print(repr(e))


def parse_args() -> str:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        "Upload compiled result to function usage dashboard, create issue"
    )
    parser.add_argument("library", help="Library to upload")
    args = parser.parse_args()
    library = args.library

    return library


if __name__ == "__main__":
    library = parse_args()

    # setup google sheet, github
    sa = gspread.service_account_from_dict(service_account)
    function_spreadsheet = sa.open(function_spreadsheet_name)
    benchmark_spreadsheet = sa.open(benchmark_spreadsheet_name)
    ivy_leaves = github.Github(ivy_leaves_token)

    extract_repo_stats(function_spreadsheet, ivy_leaves)
    upload_function_frequencies(library, function_spreadsheet)
    result_df = update_main_and_repo_sheets(function_spreadsheet)
    if os.path.exists(f"lib_scraping/compile/result/{library}/benchmark.csv"):
        upload_benchmark_result(library, benchmark_spreadsheet)

    gc_repo = ivy_leaves.get_repo("unifyai/graph-compiler")
    update_or_create_gc_issue(library, gc_repo)

    ivy_repo = ivy_leaves.get_repo("unifyai/ivy")
    update_missing_frontend_issue(result_df, ivy_repo)

    print("Uploaded:", library)
    open(f"lib_scraping/compile/result/{library}/uploaded", "w").close()
