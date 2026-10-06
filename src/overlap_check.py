"""Report word n-grams shared between paper prose and cited-paper abstracts."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def paper_prose():
    t = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    t = t.split("\\begin{document}", 1)[1].split("\\begin{thebibliography}", 1)[0]
    t = re.sub(r"%.*", " ", t)
    t = re.sub(r"\$[^$]*\$", " ", t)
    t = re.sub(r"\\(cite|ref|label|includegraphics)(\[[^\]]*\])?\{[^}]*\}", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    return t


def ngrams(ws, n):
    return {" ".join(ws[i : i + n]) for i in range(len(ws) - n + 1)}


def main():
    pw = words(paper_prose())
    p = ngrams(pw, N)
    src = (ROOT / "results" / "_src_abstracts.txt").read_text(encoding="utf-8")
    total = 0
    for block in src.split("### ")[1:]:
        title, body = block.split("\n", 1)
        hits = sorted(p & ngrams(words(body), N))
        total += len(hits)
        print(f"[{len(hits)}] {title}")
        for h in hits:
            print("    ", h)
    print(f"paper words={len(pw)}  shared {N}-grams={total}")


if __name__ == "__main__":
    main()
