"""
Download FASTQ files from ENA.
Called as a Snakemake script — accesses snakemake.* objects directly.
No subprocess calls.
"""

import gzip
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

# SNAKEMAKE BINDINGS
sample        = snakemake.params.sample
metadata_file = snakemake.params.metadata_file
output_r1     = snakemake.output.r1
output_r2     = snakemake.output.r2
log_file      = snakemake.log[0]

# CREATE LOG DIRECTORY
Path(log_file).parent.mkdir(parents=True, exist_ok=True)
Path(output_r1).parent.mkdir(parents=True, exist_ok=True)

log = open(log_file, "w")

def msg(m):
    print(m)
    log.write(m + "\n")
    log.flush()
 
def validate_gzip(path):
    try:
        with gzip.open(path, "rb") as f:
            f.read(1024)
        return True
    except Exception as e:
        msg(f"⚠️  Validation failed for {path}: {e}")
        return False


session = requests.Session()
retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))
session.mount("http://",  HTTPAdapter(max_retries=retries))

# LOAD METADATA TO MAP SAMPLE → ACCESSION
def normalize_url(link):
    if not link or str(link).lower() in ["nan", "none", ""]:
        return None
    if link.startswith("https://"):
        return link
    if link.startswith("ftp://"):
        return link.replace("ftp://", "https://", 1)
    if link.startswith("ftp."):
        return f"https://{link}"
    return f"https://ftp.sra.ebi.ac.uk/{link.lstrip('/')}"

# FIND ACCESSION PER SAMPLE
def get_fastq_links(accession):
    url = (
        "https://www.ebi.ac.uk/ena/portal/api/filereport"
        f"?accession={accession}&result=read_run&fields=fastq_ftp&format=tsv"
    )
    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return []
        lines = r.text.strip().split("\n")
        if len(lines) > 1 and lines[1].strip():
            return [normalize_url(l) for l in lines[1].split("\t")[-1].split(";")]
        return []
    except Exception as e:
        msg(f"⚠️  ENA API error: {e}")
        return []

# DOWNLOAD WITH RETRIES
def download_file(url, dest, max_retries=3):
    for attempt in range(1, max_retries + 1):
        tmp = str(dest) + ".tmp"
        try:
            msg(f"⬇️  [{attempt}/{max_retries}] {Path(dest).name}")
            r = session.get(url, stream=True, timeout=(15, 600))
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=131072):
                    if chunk:
                        fh.write(chunk)
            if not validate_gzip(tmp):
                raise ValueError("Invalid gzip")
            os.rename(tmp, dest)
            size = Path(dest).stat().st_size / 1024 / 1024
            msg(f"✅ {Path(dest).name} ({size:.1f} MB)")
            return True
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt < max_retries:
                msg(f"⚠️  Retry after 5 s: {e}")
                time.sleep(5)
            else:
                msg(f"❌ Download failed: {e}")
                return False
    return False


# MAIN
msg(f"🔍 Searching ENA links for {sample}...")
links = get_fastq_links(sample)

if not links:
    msg(f"❌ No links found for {sample}")
    log.close()
    sys.exit(1)

r1_url = next((l for l in links if l and ("_1.fastq" in l or "_R1" in l)), None)
r2_url = next((l for l in links if l and ("_2.fastq" in l or "_R2" in l)), None)

if not r1_url or not r2_url:
    msg(f"❌ Could not identify R1/R2 from: {links}")
    log.close()
    sys.exit(1)

if not download_file(r1_url, output_r1):
    log.close()
    sys.exit(1)

if not download_file(r2_url, output_r2):
    Path(output_r1).unlink(missing_ok=True)
    log.close()
    sys.exit(1)

msg(f"🎉 {sample} downloaded successfully")
log.close()
