import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


SUCCESS_CODE = "E0000000000"


class BaseClientError(RuntimeError):
    pass


class BaseClient:
    """Client for the lightweight base MPI endpoint.

    DATA_MODE=local returns mock data from MOCK_ASSETS_FILE so local tests do
    not depend on the deployed base service.
    """

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.invoke_url = os.getenv("BASE_INVOKE_URL", "").strip()
        self.data_mode = os.getenv("DATA_MODE", "local").strip().lower()
        self.timeout = float(os.getenv("BASE_INVOKE_TIMEOUT", "10"))
        default_mock = self.project_root / "web_ui" / "mock_assets.json"
        self.mock_assets_file = Path(os.getenv("MOCK_ASSETS_FILE", str(default_mock)))

    @property
    def local_mode(self) -> bool:
        return self.data_mode == "local" or not self.invoke_url

    def invoke(self, method: str, param: Any) -> Dict[str, Any]:
        if self.local_mode:
            return self._mock_invoke(method, param)

        payload = {
            "method": method,
            "content": {
                "param": param
            }
        }
        try:
            response = requests.post(self.invoke_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise BaseClientError(f"调用底座接口失败: {exc}") from exc

        code = data.get("code")
        if code not in (0, "0", SUCCESS_CODE):
            raise BaseClientError(data.get("message") or f"底座接口返回异常: {code}")
        return data

    def get_order_assets(self, order_code: str) -> List[Dict[str, Any]]:
        data = self.invoke("paas.engine.get.order.assets", {"orderCode": order_code})
        return data.get("content") or []

    def get_asset_detail(self, metano: str, side: str = "LOCAL") -> Dict[str, Any]:
        side = (side or "LOCAL").upper()
        if side == "PARTNER":
            data = self.invoke("paas.metaset.partner.detail", {"metano": metano})
        else:
            data = self.invoke("paas.metaset.detail", metano)
        return data.get("content") or {}

    def report_result(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = self.invoke("paas.engine.order.upload", payload)
        return data

    def _load_mock(self) -> Dict[str, Any]:
        if not self.mock_assets_file.exists():
            return {"orders": {}}
        with self.mock_assets_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _mock_invoke(self, method: str, param: Any) -> Dict[str, Any]:
        mock = self._load_mock()
        if method == "paas.engine.get.order.assets":
            order_code = (param or {}).get("orderCode") or os.getenv("ORDER_CODE", "ORD_LOCAL_DEMO")
            return self._ok(mock.get("orders", {}).get(order_code, []))

        if method in ("paas.metaset.detail", "paas.metaset.partner.detail"):
            metano = param if isinstance(param, str) else (param or {}).get("metano")
            detail = self._find_mock_asset(mock, metano) or {"metano": metano, "metaname": metano}
            return self._ok(detail)

        if method == "paas.engine.order.upload":
            return self._ok("success")

        raise BaseClientError(f"本地 mock 未支持 MPI: {method}")

    @staticmethod
    def _find_mock_asset(mock: Dict[str, Any], metano: Optional[str]) -> Optional[Dict[str, Any]]:
        for assets in mock.get("orders", {}).values():
            for asset in assets:
                if asset.get("metano") == metano:
                    return asset
        return None

    @staticmethod
    def _ok(content: Any) -> Dict[str, Any]:
        return {
            "code": 0,
            "message": "请求成功",
            "cause": None,
            "content": content
        }
