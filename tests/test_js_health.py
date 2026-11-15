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

from helpers import WEB, run_node
from test_js_parity import find_node, js_function

HARNESS = Path(__file__).with_name("health_harness.js")
PARTS = ("hcHeatParams", "hcHeatQuery", "hcQuery", "hcLoad", "hcLoadHeat", "hcShowHeat", "hcSaveControls",
         "hcOpenIds", "hcOpenCell", "hcOpenFiltered")
APP_PARTS = ("noFilters", "filterState", "setFilters")      # app.js — 필터 풀기(clearFilters)와 함께 쓰는 것

CELL = {"cell": {"x": 24, "y": 13, "n": 2, "ids": [5, 6], "more": 0}}
HEAT = {"grid": [48, 27], "cells": [], "max": 0, "boxes": 0, "images": 0, "sample_id": None}
REPORT = {"imgsz": 960, "default_imgsz": 640, "heatmap": HEAT}      # /api/health?heat=1 의 답(화면이 그리는 수치는 대역이 받는다)
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
    # 화면 열기 — 수치와 히트맵을 한 번에
    "load": [["load"], ["answer", REPORT]],
    "load_filter_gone": {"renderResets": True, "steps": [["load"], ["answer", REPORT], ["answer", HEAT]]},
    "load_old_server": [["load"], ["answer", {"imgsz": 960, "default_imgsz": 640}], ["answer", HEAT]],
    "load_then_heat_changed": [["load"], ["controls", {"cls": "4", "cam": ""}], ["loadheat"],
                               ["answer", REPORT], ["answer", HEAT], ["answer", HEAT]],
    # 조절 칸 남기기
    "save_before_first_answer": [["save"]],
    "save_after_sizes": [["sizes", "1280"], ["save"]],
    # 필터로 넘기기
    "filtered": [["filtered", {"camera": "cam-02"}], ["answer"]],
    "filtered_class_zero": [["filtered", {"cls": 0}], ["answer"]],
    "filtered_tag_none": [["filtered", {"tagNone": True}], ["answer"]],
    "filtered_list_fails": [["filtered", {"cls": 3}], ["fail", "서버가 꺼져 있습니다"]],
    "filtered_left_view": [["filtered", {"split": "test"}], ["leave"], ["answer"]],
    "filtered_empty_list": [["filtered", {"tag": 12}], ["answer", []]],
    "filtered_save_fails": {"saveFails": True, "steps": [["filtered", {"camera": "cam-02"}]]},
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
    app = (WEB / "app.js").read_text(encoding="utf-8")
    d = tmp_path_factory.mktemp("health")
    code = [js_function(src, name) for name in PARTS] + [js_function(app, name) for name in APP_PARTS]
    (d / "code.js").write_text("\n\n".join(code), encoding="utf-8")
    scenarios = [{"name": k, **(v if isinstance(v, dict) else {"steps": v})} for k, v in SCENARIOS.items()]
    (d / "in.json").write_text(json.dumps(scenarios), encoding="utf-8")
    done = subprocess.run([node, str(HARNESS), str(d / "code.js"), str(d / "in.json"), str(d / "out.json")],
                          capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert done.returncode == 0, f"node 실패:\n{done.stderr}"
    return json.loads((d / "out.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------- 박스 크기 칸의 이름 (서버의 경계로 만든다)

def test_size_names_and_marks_follow_the_servers_edges(tmp_path):
    """화면은 박스 크기 칸의 이름과 '32px 미만 몇 개' 를 **서버가 준 경계**(health.SIZE_EDGES)로 만든다 — 전에는 이름과 칸
    번호를 health.js 에 따로 적어, 서버의 경계를 바꾸면 옛 범위로 적었다. 칸에 넣는 것은 Python(`size_bin`), 이름을 붙이고
    더하는 것은 JS 라 둘을 견준다."""
    from autolabel import health

    node = find_node()
    if node is None:
        pytest.skip("node 가 없어 화면(JS) 쪽을 돌릴 수 없습니다")
    src = (WEB / "health.js").read_text(encoding="utf-8")
    px = [0, 3.9, 7.99, 8, 12, 15.99, 16, 20, 31.99, 32, 40, 63.9, 64, 90, 500]
    cases = {"real": list(health.SIZE_EDGES), "other": [10, 20, 40]}
    counts = {}
    for name, edges in cases.items():
        counts[name] = [0] * (len(edges) + 1)
        for v in px:
            counts[name][sum(1 for e in edges if v >= e)] += 1          # health.size_bin 과 같은 칸(경계는 위 칸)
    assert counts["real"] == [sum(1 for v in px if health.size_bin(v) == k) for k in range(len(health.SIZE_EDGES) + 1)]
    script = "\n".join([
        js_function(src, "hcSizeNames"), js_function(src, "hcUnder"),
        f"const cases = {json.dumps(cases)}, counts = {json.dumps(counts)}, out = {{}};",
        "for (const [name, edges] of Object.entries(cases)) {",
        "  out[name] = { names: hcSizeNames(edges), under: [8, 16, 32, 64, 20, 33].map((v) => hcUnder(counts[name], edges, v)) };",
        "}",
        "console.log(JSON.stringify(out));",
    ])
    got = run_node(node, script, tmp_path)
    assert got["real"]["names"] == ["8px 미만", "8–16", "16–32", "32–64", "64px 이상"]
    under = lambda limit: sum(1 for v in px if v < limit)
    assert got["real"]["under"] == [under(8), under(16), under(32), under(64), None, None]    # 경계가 아닌 값은 셀 수 없다
    assert got["other"]["names"] == ["10px 미만", "10–20", "20–40", "40px 이상"]
    assert got["other"]["under"] == [None, None, None, None, under(20), None]


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


# ----------------------------------------------------------------- 화면 열기 · 조절 칸

def test_opening_the_view_asks_once_for_numbers_and_heatmap(hc):
    """열 때는 /api/health 한 번 — 히트맵도 같은 요청으로 받는다(`heat=1`, 히트맵의 클래스·카메라 조건을 함께 싣는다).
    전에는 수치를 받은 뒤 히트맵을 따로 물어, 서버가 DB 를 두 번 읽고 덮인 몫을 두 번 쟀다."""
    got = hc["load"]
    assert got["api"] == ["/api/health?scope=all&masked=keep&imgsz=960&heat=1&cls=2"]
    assert got["heatQ"] == {"scope": "all", "masked": "keep", "cls": "2", "cam": ""}     # 칸을 누르면 이 조건으로 묻는다
    assert (got["toasts"], got["waiting"]) == ([], 0)


def test_the_heatmap_is_asked_again_only_when_the_one_that_came_does_not_fit(hc):
    # 고른 클래스가 새 범위에 없어 '전체' 로 돌아갔다 — 함께 온 히트맵(클래스 2)은 버리고 전체로 다시 받는다
    gone = hc["load_filter_gone"]
    assert gone["api"] == ["/api/health?scope=all&masked=keep&imgsz=960&heat=1&cls=2", "/api/health/heatmap?scope=all&masked=keep"]
    assert gone["heatQ"]["cls"] == ""
    # 히트맵을 싣지 않는 답(옛 서버)이면 전처럼 따로 받는다
    assert len(hc["load_old_server"]["api"]) == 2 and hc["load_old_server"]["heatQ"]["cls"] == "2"
    # 수치를 받는 사이 히트맵 조건을 바꿨다 — 함께 온 것(옛 조건)도, 먼저 물은 것도 아니라 마지막 조건으로 받은 것을 쓴다
    late = hc["load_then_heat_changed"]
    assert [a.split("?")[0] for a in late["api"]] == ["/api/health", "/api/health/heatmap", "/api/health/heatmap"]
    assert (late["heatQ"]["cls"], late["waiting"]) == ("4", 0)


def test_saving_the_controls_before_the_first_answer_keeps_the_saved_size(hc):
    """입력 크기 칸은 첫 응답 뒤에야 채워진다. 그 전에 범위를 바꿔 남기면 빈 칸(0)이 아니라 지난번 값을 그대로 둔다 —
    전에는 0 을 남겨, 다음에 열면 남겨 둔 960 대신 학습 설정값으로 돌아갔다."""
    assert hc["save_before_first_answer"]["saved"] == {"scope": "all", "masked": "keep", "imgsz": 960}
    assert hc["save_after_sizes"]["saved"] == {"scope": "all", "masked": "keep", "imgsz": 1280}


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


def test_nothing_changes_when_the_edits_in_hand_cannot_be_saved(hc):
    """라벨링에 열어 둔 장의 고치던 박스를 저장하지 못하면 필터도 목록도 보기도 그대로다 — 목록부터 바꾸면 가운데에 그 목록에
    없는 옛 이미지가 남는다(2026-11-16, app.js reloadList 와 같은 약속)."""
    got = hc["filtered_save_fails"]
    assert (got["filters"], got["listedWith"], got["views"], got["view"]) == (BEFORE, [], [], "health")
    assert (got["opening"], got["waiting"]) == (False, 0)


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
