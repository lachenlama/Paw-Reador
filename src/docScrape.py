import os
from pathlib import Path
from time import sleep
from playwright.sync_api import sync_playwright
from markdownify import markdownify as convert_html_to_md


docUrl = "https://berafarm.gitbook.io/berafarm"

output_folder = "../docs"
os.makedirs(output_folder, exist_ok=True)

def clean_filename(name):
    return "".join(c if c.isalnum() or c in "_-." else "_" for c in name).strip().replace(" ","_")


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    page.goto(docUrl)
    sleep(3)

    links = page.eval_on_selector_all(
        "aside a[href^='/berafarm']",
        "elements => elements.map(e => ({ href: e.href, title: e.textContent.trim() }))"
    )

    for link in links:
        if link['href'].startswith("/"):
            link['href'] = f"https://berafarm.gitbook.io{link['href']}"

    print(f"Found {len(links)} pages in the GitBook")

    for i, link in enumerate(links):
        print(f"Processing {i + 1}/{len(links)}: {link['title']}")
        page.goto(link['href'])
        sleep(2)  # Wait for content to load

        # Get the content inside the <main> tag
        html = page.inner_html("main")

        # Convert to Markdown
        markdown = convert_html_to_md(html)

        # Save to file
        filename = clean_filename(link["title"]) + ".txt"
        filepath = Path(output_folder) / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {link['title']}\n\n")
            f.write(markdown)

    browser.close()

print(f"\n✅ Done! Pages saved to: {output_folder}")