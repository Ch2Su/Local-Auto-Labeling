"""데이터 점검에서 이미지 보기로 넘기기 (health.js, 2026-11-15 리뷰 3·4번) — 화면 코드를 node 로 돌려 본다.

고치기 전의 두 가지:
- **히트맵 칸 누르기**가 응답을 기다리는 동안 아무것도 붙잡아 두지 않았다 — 누른 뒤 라벨링으로 가도 늦게 온 응답이 이미지
  보기로 끌고 갔고, 물을 때의 조건은 그려진 히트맵이 아니라 그 순간의 조절 칸 값이었다(다른 칸의 이미지가 열리거나 조용히 아무
  일도 없음). 두 번 누르면 두 번 물었다.
- **줄을 눌러 필터로 넘기기**(`hcOpenFiltered`)가 필터를 먼저 바꾸고 목록을 받았는데, 목록을 못 받아도 필터를 되돌리지 않았다 —
  필터 칸은 새 조건, 목록은 옛 것(`openIds` 에서 11-05 에 고친 것과 같은 종류).

`tests/health_harness.js` 가 health.js 의 `hcOpenCell`·`hcOpenIds`·`hcOpenFiltered`·`hcLoadHeat` 와 그것들이 쓰는 함수를 **그대로**
돌린다. 서버 응답·목록 받기·보기 바꾸기는 대역이고, 시나리오의 단계가 답을 하나씩 주며 그 사이에 '다른 보기로 감'·'조절 칸을
바꿈' 을 끼워 넣는다. 함수는 test_js_parity.py 와 같은 규칙으로 글자로 뗀다 — 이름이나 모양을 바꾸면 `PARTS` 도 맞출 것.

node 가 없으면 건너뛴다.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from helpers import WEB
from test_js_parity import find_node, js_function

HARNESS = Path(__file__).with_name("health_harness.js")
PARTS = ("hcHeatParams", "hcHeatQuery", "hcLoadHeat", "hcOpenIds", "hcOpenCell",
         "hcFilterState", "hcSetFilters", "hcOpenFiltered")

CELL = {"cell": {"x": 24, "y": 13, "n": 2, "ids": [5, 6], "more": 0}}
HEAT = {"grid": [48, 27], "cells": [], "max": 0, "boxes": 0, "images": 0, "sample_id": None}
# 처음 상태(health_harness.js): 왼쪽 필터 = 검수 대기 · 클래스 1 · old-cam · cctv · train · 태그 7 포함/9 제외(하나라도)
BEFORE = {"status": "pending", "classes": [1], "empty": False, "camera": "old-cam", "domain": "cctv", "split": "train",
          "tagF": {"inc": [7], "exc": [9], "any": True, "none": False},
          "selects": [["old-cam", True], ["cctv", True], ["train", True]]}
NO_TAGS = {"inc": [], "exc": [], "any": False, "none": False}

SCENARIOS = {
    # 히트맵 칸
    "cell": [["cell", 24, 13], ["answer", CELL], ["answer"]],
    "cell_left_view": [["cell", 24, 13], ["leave"], ["answer", CELL]],
    "cell_reheat": [["cell", 24, 13], ["reheat"], ["answer", CELL]],
    "cell_reheat_first": [["reheat"], ["cell", 24, 13], ["answer", CELL], ["answer"]],
    "cell_twice": [["cell", 24, 13], ["cell", 24, 13], ["answer", CELL], ["answer"]],
    "cell_fails": [["cell", 24, 13], ["fail", "응답 없음"]],
    "cell_fails_after_leaving": [["cell", 24, 13], ["leave"], ["fail", "응답 없음"]],
    "cell_gone": [["cell", 24, 13], ["answer", {"cell": {"x": 24, "y": 13, "n": 0, "ids": [], "more": 0}}]],
    "cell_then_again": [["cell", 24, 13], ["fail", "응답 없음"], ["cell", 3, 4], ["answer", CELL], ["answer"]],
    "heat_keeps_its_conditions": [["loadheat"], ["controls", {"cls": "4", "cam": "cam-09"}], ["answer", HEAT]],
    # 필터로 넘기기
    "filtered": [["filtered", {"camera": "cam-02"}], ["answer"]],
    "filtered_class_zero": [["filtered", {"cls": 0}], ["answer"]],
    "filtered_tag_none": [["filtered", {"tagNone": True}], ["answer"]],
    "filtered_list_fails": [["filtered", {"cls": 3}], ["fail", "서버가 꺼져 있습니다"]],
    "filtered_left_view": [["filtered", {"split": "test"}], ["leave"], ["answer"]],
    "filtered_empty_list": [["filtered", {"tag": 12}], ["answer", []]],
    # id 목록으로 넘기기
    "ids": [["ids", [1, 2], 3], ["answer"]],
    "ids_left_view": [["ids", [1, 2], 0], ["leave"], ["answer"]],
    "ids_fail": [["ids", [1, 2], 0], ["fail", "응답 없음"]],
}


@pytest.fixture(scope="module")
def hc(tmp_path_factory) -> dict:
    node = find_node()
    if node is None:
        pytest.skip("node 가 없어 화면(JS) 쪽을 돌릴 수 없습니다")
    src = (WEB / "health.js").read_text(encoding="utf-8")
    d = tmp_path_factory.mktemp("health")
    (d / "code.js").write_text("\n\n".join(js_function(src, name) for name in PARTS), encoding="utf-8")
    (d / "in.json").write_text(json.dumps([{"name": k, "steps": v} for k, v in SCENARIOS.items()]), encoding="utf-8")
    done = subprocess.run([node, str(HARNESS), str(d / "code.js"), str(d / "in.json"), str(d / "out.json")],
                          capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert done.returncode == 0, f"node 실패:\n{done.stderr}"
    return json.loads((d / "out.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------- 히트맵 칸

def test_a_cell_asks_with_the_conditions_of_the_heatmap_on_screen(hc):
    """조절 칸(범위 all·넣고 셈·클래스 2)이 그려진 히트맵(승인·빼고 셈·cam-01)과 달라도, 칸은 **그려진 것**의 조건으로 묻는다."""
    got = hc["cell"]
    assert got["api"] == ["/api/health/heatmap?scope=approved&masked=drop&camera=cam-01&cell=24%2C13"]
    assert (got["opened"], got["views"], got["view"]) == ([[5, 6]], ["gallery"], "gallery")
    assert (got["toasts"], got["opening"], got["waiting"]) == ([], False, 0)


def test_a_late_cell_answer_does_not_pull_you_back(hc):
    """누른 뒤 다른 보기로 갔으면 늦게 온 응답은 버린다 — 전에는 라벨링으로 간 사람을 이미지 보기로 끌고 갔다."""
    got = hc["cell_left_view"]
    assert (got["opened"], got["views"], got["view"], got["toasts"]) == ([], [], "label", [])
    assert got["opening"] is False


def test_a_cell_answer_is_dropped_once_the_heatmap_is_being_reloaded(hc):
    """누른 뒤 조절 칸을 바꿔 히트맵을 다시 받기 시작했으면 버린다. 이미 받는 중일 때 누른 것은 눈에 보이는 칸이라 연다."""
    assert (hc["cell_reheat"]["opened"], hc["cell_reheat"]["views"]) == ([], [])
    assert (hc["cell_reheat_first"]["opened"], hc["cell_reheat_first"]["views"]) == ([[5, 6]], ["gallery"])


def test_two_clicks_ask_once(hc):
    got = hc["cell_twice"]
    assert (len(got["api"]), got["opened"], got["views"]) == (1, [[5, 6]], ["gallery"])


def test_a_failed_cell_request_says_so_and_frees_the_next_click(hc):
    got = hc["cell_fails"]
    assert got["toasts"] == [["그 칸의 이미지를 받지 못했습니다: 응답 없음", True]]
    assert (got["opened"], got["opening"]) == ([], False)
    assert hc["cell_fails_after_leaving"]["toasts"] == []          # 떠난 뒤의 실패는 알릴 것이 없다
    again = hc["cell_then_again"]                                  # 실패한 뒤에도 다음 누르기는 먹는다
    assert (len(again["api"]), again["opened"], again["views"]) == (2, [[5, 6]], ["gallery"])
    assert again["api"][1].endswith("cell=3%2C4")


def test_a_cell_whose_boxes_are_gone_says_so(hc):
    """그린 뒤에 박스가 없어진 칸 — 전에는 아무 일도 없었다."""
    got = hc["cell_gone"]
    assert (got["opened"], got["views"]) == ([], [])
    assert len(got["toasts"]) == 1 and "없어졌습니다" in got["toasts"][0][0]


def test_the_heatmap_remembers_the_conditions_it_was_asked_with(hc):
    """받는 사이 조절 칸을 또 바꿔도, 그려진 히트맵의 조건(H.heatQ)은 **물었을 때**의 값이다."""
    got = hc["heat_keeps_its_conditions"]
    assert got["api"] == ["/api/health/heatmap?scope=all&masked=keep&cls=2"]
    assert got["heatQ"] == {"scope": "all", "masked": "keep", "cls": "2", "cam": ""}


# ----------------------------------------------------------------- 필터로 넘기기

def test_a_row_sets_that_one_filter_and_opens_the_gallery(hc):
    got = hc["filtered"]
    want = {"status": "live", "classes": [], "empty": False, "camera": "cam-02", "domain": "", "split": "",
            "tagF": NO_TAGS, "selects": [["cam-02", True], ["", False], ["", False]]}
    assert got["listedWith"] == [want] and got["filters"] == want    # 목록을 받을 때 이미 새 조건이다
    assert (got["loaded"], got["views"], got["view"], got["toasts"]) == ([42], ["gallery"], "gallery", [])
    assert hc["filtered_class_zero"]["filters"]["classes"] == [0]    # 클래스 0 도 '없음' 으로 읽지 않는다
    assert hc["filtered_tag_none"]["filters"]["tagF"] == {**NO_TAGS, "none": True}
    empty = hc["filtered_empty_list"]                                # 맞는 이미지가 없으면 첫 장을 열지 않고 넘어간다
    assert (empty["loaded"], empty["views"], empty["filters"]["tagF"]["inc"]) == ([], ["gallery"], [12])


def test_filters_go_back_when_the_list_cannot_be_fetched(hc):
    """목록을 못 받으면 필터를 앞 상태로 — 필터 칸은 새 조건인데 목록은 옛 것인 채로 남지 않는다."""
    got = hc["filtered_list_fails"]
    assert got["listedWith"][0]["classes"] == [3] and got["listedWith"][0]["status"] == "live"   # 물을 때는 새 조건이었다
    assert got["filters"] == BEFORE                              # 상태 칸·클래스·카메라·도메인·분할·태그·칸의 표시까지
    assert got["syncs"] == 2                                     # 받기 전 한 번(새 조건) + 되돌린 뒤 한 번 — 칸 잠금·요약
    assert (got["views"], got["view"], got["loaded"]) == ([], "health", [])
    assert got["toasts"] == [["목록을 받지 못했습니다: 서버가 꺼져 있습니다", True]]
    assert got["opening"] is False


def test_a_late_list_does_not_pull_you_back_either(hc):
    """목록을 받는 사이 다른 보기로 갔으면 보기는 바꾸지 않는다(필터와 목록은 새 조건 그대로)."""
    got = hc["filtered_left_view"]
    assert (got["views"], got["view"], got["filters"]["split"], got["loaded"]) == ([], "label", "test", [42])
    got = hc["ids_left_view"]
    assert (got["opened"], got["views"], got["view"]) == ([[1, 2]], [], "label")


def test_an_id_list_opens_the_gallery_and_reports_what_was_cut(hc):
    got = hc["ids"]
    assert (got["opened"], got["views"]) == ([[1, 2]], ["gallery"])
    assert got["toasts"] == [["앞의 2장만 열었습니다 (3장 더 있습니다)", False]]
    failed = hc["ids_fail"]
    assert (failed["views"], failed["opening"]) == ([], False)
    assert failed["toasts"] == [["목록으로 열지 못했습니다: 응답 없음", True]]
