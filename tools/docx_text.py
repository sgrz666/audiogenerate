"""Extract plain text from a .docx using only the stdlib (read-only)."""
import re
import sys
import zipfile


def main():
    path = sys.argv[1]
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "replace")
    # paragraph and table-cell boundaries -> newlines
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"</w:tc>", " | ", xml)
    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    xml = re.sub(r"<w:br[^>]*/>", "\n", xml)
    text = re.sub(r"<[^>]+>", "", xml)
    text = (text.replace("&amp;", "&").replace("&lt;", "<")
                .replace("&gt;", ">").replace("&quot;", '"')
                .replace("&apos;", "'"))
    lines = [ln.strip() for ln in text.split("\n")]
    out = "\n".join(ln for ln in lines if ln)
    sys.stdout.write(out + "\n")


if __name__ == "__main__":
    main()