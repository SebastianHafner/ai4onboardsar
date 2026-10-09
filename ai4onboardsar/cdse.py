"""Minimal Copernicus Data Space Ecosystem (CDSE) client: auth, search, download."""
import os
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1/Products({})/$value"


class CDSE:
    def __init__(self, env_file: str | None = None):
        load_dotenv(env_file or Path(__file__).resolve().parents[1] / ".env")
        self.user = os.environ["CDSE_USERNAME"]
        self.password = os.environ["CDSE_PASSWORD"]
        self._token = None
        self._token_time = 0.0

    @property
    def token(self) -> str:
        # Access tokens are valid for 10 min; refresh conservatively.
        if self._token is None or time.time() - self._token_time > 480:
            r = requests.post(TOKEN_URL, data={
                "client_id": "cdse-public", "grant_type": "password",
                "username": self.user, "password": self.password})
            r.raise_for_status()
            self._token = r.json()["access_token"]
            self._token_time = time.time()
        return self._token

    def search(self, odata_filter: str, top: int = 100) -> list[dict]:
        r = requests.get(CATALOGUE, params={
            "$filter": odata_filter, "$top": top,
            "$select": "Id,Name,ContentLength,ContentDate"})
        r.raise_for_status()
        return r.json()["value"]

    def by_name(self, name: str) -> dict:
        res = self.search(f"Name eq '{name}'")
        if not res:
            raise ValueError(f"Product not found: {name}")
        return res[0]

    def matching_raw(self, grd_name: str) -> list[dict]:
        """RAW (L0) products of the same datatake overlapping the GRD sensing time."""
        p = grd_name.replace(".SAFE", "").split("_")
        mission, mode, start, stop, datatake = p[0], p[1], p[4], p[5], p[7]
        fmt = "%Y%m%dT%H%M%S"
        t0, t1 = datetime.strptime(start, fmt), datetime.strptime(stop, fmt)
        cands = self.search(
            f"startswith(Name,'{mission}_{mode}_RAW__0S') and contains(Name,'_{datatake}_') "
            f"and ContentDate/Start lt {(t1 + timedelta(seconds=1)).isoformat()}Z "
            f"and ContentDate/End gt {(t0 - timedelta(seconds=1)).isoformat()}Z")
        return sorted(cands, key=lambda c: c["Name"])

    def download(self, product: dict, out_dir: Path, unzip: bool = True) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = out_dir / product["Name"]
        if safe.exists():
            return safe
        zpath = out_dir / (product["Name"] + ".zip")
        tmp = zpath.with_suffix(".part")
        if not zpath.exists():
            # Follow redirects manually so the auth header is kept.
            url = DOWNLOAD.format(product["Id"])
            sess = requests.Session()
            sess.headers["Authorization"] = f"Bearer {self.token}"
            r = sess.get(url, allow_redirects=False)
            while r.status_code in (301, 302, 303, 307):
                url = r.headers["Location"]
                r = sess.get(url, allow_redirects=False)
            r = sess.get(url, stream=True)
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 << 20):
                    f.write(chunk)
            tmp.rename(zpath)
        if unzip:
            with zipfile.ZipFile(zpath) as z:
                z.extractall(out_dir)
            zpath.unlink()
            return safe
        return zpath
