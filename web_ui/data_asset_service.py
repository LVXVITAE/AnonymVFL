import csv
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from web_ui.base_client import BaseClient


METANO_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class DataAssetError(ValueError):
    pass


class DataAssetService:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.client = BaseClient(self.project_root)
        default_local_dir = self.project_root / "web_ui" / "mock_assets"
        self.local_asset_dir = Path(os.getenv("LOCAL_ASSET_DIR", str(default_local_dir)))
        self.nfs_mount_path = Path(os.getenv("NFS_MOUNT_PATH", "/data/nfs"))
        self.data_mode = os.getenv("DATA_MODE", "local").strip().lower()

    def asset_base_dir(self) -> Path:
        return self.local_asset_dir if self.data_mode == "local" else self.nfs_mount_path

    def asset_path(self, metano: str) -> Path:
        metano = self._validate_metano(metano)
        return self.asset_base_dir() / f"{metano}.csv"

    def path_payload(self, metano: str) -> Dict[str, Any]:
        path = self.asset_path(metano)
        return {
            "status": "success",
            "metano": metano,
            "path": str(path),
            "exists": path.exists(),
            "data_mode": self.data_mode
        }

    def order_assets(self, order_code: Optional[str]) -> List[Dict[str, Any]]:
        order_code = order_code or os.getenv("ORDER_CODE", "ORD_LOCAL_DEMO")
        assets = self.client.get_order_assets(order_code)
        return [self._attach_asset_runtime_fields(asset) for asset in assets]

    def asset_detail(self, metano: str, side: str = "LOCAL") -> Dict[str, Any]:
        detail = self.client.get_asset_detail(metano, side=side)
        detail = dict(detail)
        detail.setdefault("metano", metano)
        detail = self._attach_asset_runtime_fields(detail)
        csv_meta = self.csv_metadata(metano)
        if csv_meta:
            detail.setdefault("line_count", csv_meta["line_count"])
            detail.setdefault("columns", csv_meta["columns"])
        return detail

    def report_result(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {
            "orderCode": payload.get("orderCode") or os.getenv("ORDER_CODE", ""),
            "orderType": payload.get("orderType", "OFFLINE_TASK"),
            "extra": payload.get("extra") or {},
            "output": payload.get("output") or {}
        }
        if "resultPath" in payload and "resultPath" not in normalized["output"]:
            normalized["output"]["resultPath"] = payload["resultPath"]
        return self.client.report_result(normalized)

    def resolve_dataset_value(self, value: Optional[str], metano: Optional[str] = None) -> Optional[str]:
        if metano:
            return str(self.asset_path(metano))
        if not value:
            return None
        if value.startswith("asset://"):
            return str(self.asset_path(value[len("asset://"):]))
        return value

    def dataset_options_for_side(self, side: str) -> List[Dict[str, Any]]:
        order_code = os.getenv("ORDER_CODE", "ORD_LOCAL_DEMO")
        side = side.upper()
        options = []
        for asset in self.order_assets(order_code):
            if (asset.get("assetSide") or "").upper() != side:
                continue
            metano = asset.get("metano")
            if not metano:
                continue
            path = self.asset_path(metano)
            options.append({
                "name": f"{asset.get('metaname') or metano} ({metano})",
                "path": f"asset://{metano}",
                "resolved_path": str(path),
                "metano": metano,
                "assetSide": side,
                "size": path.stat().st_size if path.exists() else 0,
                "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else None,
                "exists": path.exists()
            })
        return options

    def csv_metadata(self, metano: str, sample_size: int = 20) -> Dict[str, Any]:
        path = self.asset_path(metano)
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = []
            line_count = 0
            for row in reader:
                line_count += 1
                if len(rows) < sample_size:
                    rows.append(row)
        return {
            "line_count": line_count,
            "columns": [
                {
                    "name": name,
                    "alias": "",
                    "kind": self._infer_kind([row.get(name, "") for row in rows]),
                    "comment": "",
                    "flags": 1 if name.lower() == "id" else 0
                }
                for name in fieldnames
            ]
        }

    def _attach_asset_runtime_fields(self, asset: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(asset)
        metano = result.get("metano")
        if metano:
            path = self.asset_path(metano)
            result["path"] = str(path)
            result["assetUri"] = f"asset://{metano}"
            result["exists"] = path.exists()
        return result

    @staticmethod
    def _validate_metano(metano: str) -> str:
        if not metano or not METANO_RE.match(metano):
            raise DataAssetError("数据资产编号只能包含字母、数字、下划线、横线和点")
        return metano

    @staticmethod
    def _infer_kind(values: List[str]) -> str:
        clean = [v for v in values if v not in ("", None)]
        if not clean:
            return "String"
        try:
            for value in clean:
                int(str(value))
            return "Long"
        except ValueError:
            pass
        try:
            for value in clean:
                float(str(value))
            return "Double"
        except ValueError:
            return "String"
