import os
import re
from pathlib import Path

# list all workflow files
workflow_dir = ".github/workflows"
workflow_files = os.listdir(workflow_dir)

# generate the markdown for the badges
base_url = "https://github.com/huggingface/optimum-benchmark/actions/workflows"
api_badges = []
cli_badges = []
for file in workflow_files:
    # extract the name from the file name
    name = re.sub(r"(test_|\.yaml)", "", file).upper()
    badge_url = f"{base_url}/{file}/badge.svg"
    workflow_url = f"{base_url}/{file}"
    badge = f"[![{name}]({badge_url})]({workflow_url})"
    if "api" in file:
        api_badges.append(badge)
    elif "cli" in file:
        cli_badges.append(badge)

# order the badges
api_badges = sorted(api_badges)
cli_badges = sorted(cli_badges)

# read the README file
readme_path = Path("README.md")
readme_text = readme_path.read_text()

# find the position to insert the badges
api_start_pos = readme_text.index("### API 📈") + len("### API 📈\n\n")
api_end_pos = readme_text.index("#", api_start_pos)
cli_start_pos = readme_text.index("### CLI 📈") + len("### CLI 📈\n\n")
cli_end_pos = readme_text.index("#", cli_start_pos)

# insert the badges into the README text
new_readme_text = (
    readme_text[:api_start_pos]
    + "\n".join(api_badges)
    + "\n\n"
    + readme_text[api_end_pos:cli_start_pos]
    + "\n".join(cli_badges)
    + "\n\n"
    + readme_text[cli_end_pos:]
)

# write the new README text to the file
readme_path.write_text(new_readme_text)
