# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
"""例外。"""


class ParcelError(RuntimeError):
    """PARCEL の呼び出しが失敗した。全ランクで同じ例外が投げられる。"""

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class NotBuiltError(ParcelError):
    """要求したソルバがネイティブ層のビルドから外されている。"""
