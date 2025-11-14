import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "settings.json"

with CONFIG_PATH.open(encoding="utf-8") as f:
    _CFG = json.load(f)


def get_config():
    """전체 설정 딕셔너리 리턴"""
    return _CFG


def get_db_path():
    """DB 파일 경로(Path 객체) 리턴"""
    return ROOT / _CFG["database"]["path"]