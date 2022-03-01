from bs4 import BeautifulSoup
import re, requests
from tqdm import tqdm

def scrapeAbstract(url):
    # Spoof User Agent to avoid being blocked by sites
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.0'
    }
    try:
        page = requests.get(url, headers=headers)
    except requests.exceptions.RequestException:
        # Error in requesting the page
        return False
    if page.status_code != 200:
        # Could not retrieve page
        return False
    soup = BeautifulSoup(page.content, 'html.parser')
    try:
        for t in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "div", "p", "strong", "td"]):
            if re.search(r"^\s?Abstract(:?)\s?$", t.text):
                if "Abstract" in t.text:
                    # Strategy: look for siblings with text length > 30 words = 150 letters (average english word has 5 letters)
                    if len(t.nextSibling.text) > 150:
                        # search in next sibling
                        return t.nextSibling.text
                    elif len(t.nextSibling.nextSibling.text) > 150:
                        # second in second next sibling
                        return t.nextSibling.nextSibling.text
                    elif len(t.nextSibling.findChildren('p')[0].text) > 150:
                        # for science direct: abstract is in first p tag in next sibling
                        return t.nextSibling.findChildren('p')[0].text
                    else:
                        return False
            
        # Term "Abstract" not found on page in vicinity of large text fragements
        if url.find("sciencedirect") > 1:
        # structure for Sciencedirect pages:
            return soup.find(class_="author-highlights").nextSibling.nextSibling.text.replace("\n", "")
        else:
            # Structure for IEEEXPLORE, ARXIV
            desc = soup.find("meta",  property="og:description")
            if len(desc["content"]) > 150:
                return desc["content"].replace("\n", " ")
            else:
                return False
    except Exception:
        # Something went wrong during parsing, exit
        return False        

if __name__ == "__main__":
    fileNameIn = r"C:\Users\flietz\OneDrive - TU Wien\!Studium\1_MSc\!Diplomarbeit\SoA\LPPM\test.txt"
    fileNameOut = r"C:\Users\flietz\OneDrive - TU Wien\!Studium\1_MSc\!Diplomarbeit\SoA\LPPM\test_out.txt"
    fileNameLog = r"C:\Users\flietz\OneDrive - TU Wien\!Studium\1_MSc\!Diplomarbeit\SoA\LPPM\test_log.txt"

    # read file, find url, scrap abstract and overwrite abstract line in file
    content = []
    errors = []
    with open(fileNameIn, "r", encoding='utf-8-sig') as f:
        # RIS-files from EndNote contain a BOM and therefore these bytes shall be read as file info rather than as strings
            temp = []
            for l in f.readlines():
                if l != "\n":
                    temp.append(l)
                else:
                    content.append(temp)
                    temp = []
           
    for block in tqdm(content):
        noUrlFlag = False
        noAbstractFlag = False
        try:
            url = [e.replace("UR  -", "").replace("\n", "").strip() for e in block if e.startswith("UR  - ")][0]
        except IndexError:
            noUrlFlag = True
        try:
            abstractIdx = [i for i,x in enumerate(block) if x.startswith("AB  -")][0]
        except IndexError:
            noAbstractFlag = True
        if len(block[abstractIdx]) < 220 and not noUrlFlag and not noAbstractFlag:
            # only scrape abstract if no full abstract is available (i.e., shortened with ...) and if URL is available
            try:
                res = scrapeAbstract(url)
            except TypeError:
                res = False
            if res:
                # if no result could be obtained, leave shortened (...) abstract as it is
                block[abstractIdx] = f"AB  - {res}"
            else:
                # no abstract was attained, write corresponding URL to error log
                if url:
                    errors.append(f"{url}\n")
                else:
                    errors.append("Could not retrieve abstract, no url available.")
            # write errors
            with open(fileNameLog, "a+", encoding='utf-8-sig') as fe:
                fe.writelines(errors)
                fe.write('\n')
                errors = []
        # write everything to file
        with open(fileNameOut, "a+", encoding='utf-8-sig') as fw:
            fw.writelines(block)
            fw.write('\n')
            
        