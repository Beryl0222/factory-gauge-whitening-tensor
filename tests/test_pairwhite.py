"""成对统计与分组白化发布的核对测试。"""
import json
import unittest
from pathlib import Path

import numpy as np

from caseio import load_case
from pairwhite import GroupSpec, Pipeline, SubmitOutcome, run_case
from pairwhite.publisher import group_pairs

DATA = Path(__file__).resolve().parents[1] / "data"


def make_rows(tensors, keys):
    """把数据集张量展开成 (x行, mask行, w) 列表。"""
    rows = []
    for ds in keys:
        x, m, w = tensors[ds + "_x"], tensors[ds + "_mask"], tensors[ds + "_w"]
        for r in range(len(w)):
            rows.append((np.asarray(x[r], float), np.asarray(m[r], bool), float(w[r])))
    return rows


def naive_pair(rows, i, j):
    """朴素配对统计：只看两侧共有效的正权样本，加权总体口径。"""
    sel = [(x, w) for x, m, w in rows if w > 0 and m[i] and m[j]]
    if not sel:
        return None
    W = np.array([w for _, w in sel])
    X = np.array([x[i] for x, _ in sel])
    Y = np.array([x[j] for x, _ in sel])
    sw = W.sum()
    mx = W @ X / sw
    my = W @ Y / sw
    return {
        "sum_w": sw,
        "n": len(sel),
        "n_eff": sw * sw / (W @ W),
        "mean_i": mx,
        "mean_j": my,
        "cov": (W * (X - mx) * (Y - my)).sum() / sw,
    }


def naive_channel_mean(rows, c):
    sel = [(x, w) for x, m, w in rows if w > 0 and m[c]]
    W = np.array([w for _, w in sel])
    X = np.array([x[c] for x, _ in sel])
    return W @ X / W.sum()


def load_case_data():
    return load_case(DATA / "capture.json")


class PairwiseStatsTests(unittest.TestCase):
    def setUp(self):
        self.records, self.tensors = load_case_data()

    def build_pipeline(self):
        return Pipeline.from_handoff(self.records)

    def submit_log(self, pipe):
        for sub in self.records["submissions"]:
            ds = sub["dataset"]
            pipe.submit(sub["id"], sub["rev"], self.tensors[ds + "_x"],
                        self.tensors[ds + "_mask"], self.tensors[ds + "_w"])

    def test_pairwise_stats_match_naive(self):
        pipe = self.build_pipeline()
        self.submit_log(pipe)
        rows = make_rows(self.tensors, ["a", "c"])  # b 已被更正撤销
        for g in pipe.groups:
            for i, j in group_pairs(g.columns):
                want = naive_pair(rows, i, j)
                got = pipe.accumulator.pair_summary(i, j)
                self.assertTrue(got.has_estimate)
                self.assertAlmostEqual(got.n_eff, want["n_eff"], places=12)
                self.assertAlmostEqual(got.mean_i, want["mean_i"], places=12)
                self.assertAlmostEqual(got.mean_j, want["mean_j"], places=12)
                self.assertAlmostEqual(got.cov, want["cov"], places=12)
                self.assertEqual(got.n, want["n"])
        for c in range(5):
            got = pipe.accumulator.channel_summary(c)
            self.assertAlmostEqual(got.mean, naive_channel_mean(rows, c), places=12)

    def test_resend_is_idempotent(self):
        t = self.tensors
        once = self.build_pipeline()
        once.submit("p2", 1, t["b_x"], t["b_mask"], t["b_w"])
        twice = self.build_pipeline()
        twice.submit("p2", 1, t["b_x"], t["b_mask"], t["b_w"])
        outcome = twice.submit("p2", 1, t["b_x"], t["b_mask"], t["b_w"])
        self.assertEqual(outcome, SubmitOutcome.DUPLICATE)
        np.testing.assert_array_equal(twice.accumulator.pair_stats, once.accumulator.pair_stats)
        np.testing.assert_array_equal(twice.accumulator.channel_stats, once.accumulator.channel_stats)
        self.assertEqual(twice.accumulator.live_segments, 1)

    def test_correction_retracts_old_and_new_pairs(self):
        t = self.tensors
        corrected = self.build_pipeline()
        corrected.submit("p1", 1, t["a_x"], t["a_mask"], t["a_w"])
        corrected.submit("p2", 1, t["b_x"], t["b_mask"], t["b_w"])
        outcome = corrected.submit("p2", 2, t["c_x"], t["c_mask"], t["c_w"])
        self.assertEqual(outcome, SubmitOutcome.REPLACED)
        fresh = self.build_pipeline()
        fresh.submit("p1", 1, t["a_x"], t["a_mask"], t["a_w"])
        fresh.submit("p2", 2, t["c_x"], t["c_mask"], t["c_w"])
        np.testing.assert_allclose(corrected.accumulator.pair_stats,
                                   fresh.accumulator.pair_stats, atol=1e-12)
        np.testing.assert_allclose(corrected.accumulator.channel_stats,
                                   fresh.accumulator.channel_stats, atol=1e-12)
        # 更正确实改变了涉及 depth 通道的配对（b 与 c 的第 2 列有效位不同）
        only_b = self.build_pipeline()
        only_b.submit("p2", 1, t["b_x"], t["b_mask"], t["b_w"])
        self.assertNotAlmostEqual(only_b.accumulator.pair_summary(0, 2).sum_w,
                                  fresh.accumulator.pair_summary(0, 2).sum_w)

    def test_sharding_invariance(self):
        t = self.tensors
        rows = make_rows(t, ["a", "c"])

        def submit_rows(pipe, seg_id, chunk):
            x = np.array([r[0] for r in chunk])
            m = np.array([r[1] for r in chunk])
            w = np.array([r[2] for r in chunk])
            pipe.submit(seg_id, 1, x, m, w)

        whole = self.build_pipeline()
        submit_rows(whole, "all", rows)
        # 不同分片、不同提交顺序，同样的权重与值
        sharded = self.build_pipeline()
        chunks = [rows[5:], rows[:2], rows[2:5]]
        for k, chunk in enumerate(chunks):
            submit_rows(sharded, f"part{k}", chunk)
        np.testing.assert_allclose(sharded.accumulator.pair_stats,
                                   whole.accumulator.pair_stats, atol=1e-9)
        np.testing.assert_allclose(sharded.accumulator.channel_stats,
                                   whole.accumulator.channel_stats, atol=1e-9)
        d1 = {k: (v.decision, v.version) for k, v in whole.publish().items()}
        d2 = {k: (v.decision, v.version) for k, v in sharded.publish().items()}
        self.assertEqual(d1, d2)

    def test_zero_weight_rows_have_no_contribution(self):
        pipe = self.build_pipeline()
        before = pipe.accumulator.pair_stats.copy()
        x = np.array([[1e6, 1e6, 1e6, 1e6, 1e6]])
        m = np.ones((1, 5), dtype=bool)
        pipe.submit("zero", 1, x, m, np.array([0.0]))
        np.testing.assert_array_equal(pipe.accumulator.pair_stats, before)
        np.testing.assert_array_equal(pipe.accumulator.channel_stats,
                                      np.zeros_like(pipe.accumulator.channel_stats))

    def test_pairwise_means_not_single_channel_means(self):
        # 单通道均值与配对均值不同；代入单通道均值会改变协方差
        x = np.array([[0.0, 100.0], [2.0, -100.0], [10.0, 0.0], [0.0, 50.0]])
        m = np.array([[1, 1], [1, 1], [1, 0], [0, 1]], dtype=bool)
        w = np.ones(4)
        pipe = Pipeline(["u", "v"], [GroupSpec("g", (0, 1), 0.5, 100.0, 100.0)])
        pipe.submit("s", 1, x, m, w)
        got = pipe.accumulator.pair_summary(0, 1)
        self.assertAlmostEqual(got.mean_i, 1.0, places=12)
        self.assertAlmostEqual(got.mean_j, 0.0, places=12)
        self.assertAlmostEqual(got.cov, -100.0, places=12)
        m0 = pipe.accumulator.channel_summary(0).mean
        m1 = pipe.accumulator.channel_summary(1).mean
        wrong = ((0 - m0) * (100 - m1) + (2 - m0) * (-100 - m1)) / 2
        self.assertNotAlmostEqual(got.cov, wrong, places=6)

    def test_state_size_independent_of_sample_count(self):
        rng = np.random.default_rng(3)
        pipe = self.build_pipeline()
        shape_before = pipe.accumulator.pair_stats.shape
        for k in range(3):
            x = rng.normal(size=(2000, 5))
            m = np.ones((2000, 5), dtype=bool)
            w = rng.random(2000) + 0.1
            pipe.submit(f"bulk{k}", 1, x, m, w)
        # 充分统计量与配对状态不随样本数增长，只有存活片段贡献线性增加
        self.assertEqual(pipe.accumulator.pair_stats.shape, shape_before)
        self.assertEqual(pipe.accumulator.live_segments, 3)

    def test_conflicting_resubmission_raises(self):
        t = self.tensors
        pipe = self.build_pipeline()
        pipe.submit("p", 1, t["a_x"], t["a_mask"], t["a_w"])
        with self.assertRaises(ValueError):
            pipe.submit("p", 1, t["b_x"], t["b_mask"], t["b_w"])
        # 同内容仅修订号变化：不重复计入
        outcome = pipe.submit("p", 2, t["a_x"], t["a_mask"], t["a_w"])
        self.assertEqual(outcome, SubmitOutcome.DUPLICATE)
        self.assertEqual(pipe.accumulator.live_segments, 1)

    def test_large_fragment_is_chunked(self):
        rng = np.random.default_rng(7)
        n, c = 600, 5
        x = rng.normal(size=(n, c))
        m = rng.random((n, c)) > 0.25
        m[:, 0] = True
        w = rng.random(n) + 0.1
        pipe = self.build_pipeline()
        pipe.submit("big", 1, x, m, w)
        rows = [(x[r], m[r], float(w[r])) for r in range(n)]
        for i, j in pipe.accumulator.pairs:
            want = naive_pair(rows, i, j)
            got = pipe.accumulator.pair_summary(i, j)
            self.assertAlmostEqual(got.cov, want["cov"], places=9)
            self.assertAlmostEqual(got.n_eff, want["n_eff"], places=9)


class PublishTests(unittest.TestCase):
    def test_missing_pair_blocks_without_zerofill(self):
        # 两个通道从未共测：覆盖不足，不能填零宣称验证完成
        x = np.array([[1.0, 0.0], [0.0, 2.0]])
        m = np.array([[1, 0], [0, 1]], dtype=bool)
        w = np.ones(2)
        pipe = Pipeline(["u", "v"], [GroupSpec("g", (0, 1), 1.0, 10.0, 10.0)])
        pipe.submit("s", 1, x, m, w)
        decision = pipe.publish()["g"]
        self.assertEqual(decision.decision, "unavailable")
        self.assertIsNone(decision.version)
        self.assertEqual(decision.block["reason"], "coverage")
        self.assertEqual(decision.block["detail"]["pairs_without_estimate"], [[0, 1]])
        self.assertIsNone(pipe.publisher.published("g"))
        report = pipe.report()
        self.assertIsNone(report["pairs"]["g"]["u~v"])
        self.assertFalse(report["groups"]["g"]["consumable"])
        skipped = pipe.transform([1.0, 2.0])["skipped"]
        self.assertIn("g", skipped)

    def test_coverage_uses_min_pair_including_diagonal(self):
        x = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        m = np.array([[1, 1], [0, 1], [0, 1]], dtype=bool)
        w = np.ones(3)
        passing = Pipeline(["u", "v"], [GroupSpec("g", (0, 1), 1.0, 10.0, 10.0)])
        passing.submit("s", 1, x, m, w)
        self.assertEqual(passing.publish()["g"].decision, "published")  # 边界 >= 通过
        blocking = Pipeline(["u", "v"], [GroupSpec("g", (0, 1), 1.5, 10.0, 10.0)])
        blocking.submit("s", 1, x, m, w)
        decision = blocking.publish()["g"]
        self.assertEqual(decision.decision, "unavailable")
        self.assertAlmostEqual(decision.coverage, 1.0, places=12)  # 对角线 u~u 拉低覆盖

    def indefinite_pipeline(self, max_adjust_fro):
        # 三通道两两在不同样本上测量，配对协方差出现负特征值
        groups = [GroupSpec("g", (0, 1, 2), 3.0, 2.0, max_adjust_fro)]
        pipe = Pipeline(["p", "q", "r"], groups)
        pipe.submit("s1", 1,
                    np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0]]),
                    np.array([[1, 1, 0]] * 3, dtype=bool), np.ones(3))
        pipe.submit("s2", 1,
                    np.array([[1.0, 0.0, 1.0], [2.0, 0.0, 2.0], [3.0, 0.0, 3.0]]),
                    np.array([[1, 0, 1]] * 3, dtype=bool), np.ones(3))
        pipe.submit("s3", 1,
                    np.array([[0.0, 1.0, -1.0], [0.0, 2.0, -2.0], [0.0, 3.0, -3.0]]),
                    np.array([[0, 1, 1]] * 3, dtype=bool), np.ones(3))
        return pipe

    def test_indefinite_covariance_is_corrected(self):
        pipe = self.indefinite_pipeline(max_adjust_fro=1.0)
        decision = pipe.publish()["g"]
        self.assertEqual(decision.decision, "published")
        pub = pipe.publisher.published("g")
        # 与累积器导出的协方差对照修正量
        cov = np.array([[pipe.accumulator.pair_summary(i, j).cov for j in (0, 1, 2)]
                        for i in (0, 1, 2)])
        evals = np.linalg.eigvalsh(cov)
        self.assertLess(evals[0], 0.0)  # 构造的数据确实不定
        floor = 1.0 / 4.0  # 1 / max_norm^2
        want_adjust = float(np.sqrt(((np.maximum(evals, floor) - evals) ** 2).sum()))
        self.assertAlmostEqual(pub.adjust_fro, want_adjust, places=12)
        self.assertLessEqual(pub.adjust_fro, 1.0)
        # 交付的是修正后正定协方差的对称逆平方根
        w = pub.whitening
        np.testing.assert_allclose(w, w.T, atol=1e-12)
        c_adj = np.linalg.inv(w @ w)
        self.assertGreaterEqual(np.linalg.eigvalsh(c_adj).min(), floor - 1e-9)
        np.testing.assert_allclose(w @ c_adj @ w, np.eye(3), atol=1e-9)
        self.assertLessEqual(pub.spectral_norm, 2.0)
        self.assertAlmostEqual(pub.spectral_norm, 1.0 / np.sqrt(np.linalg.eigvalsh(c_adj).min()),
                               places=9)

    def test_conflicting_limits_keep_previous_version(self):
        groups = [
            GroupSpec("g", (0, 1, 2), 1.0, 2.0, 0.5),
            GroupSpec("h", (3, 4), 1.0, 10.0, 5.0),
        ]
        pipe = Pipeline(["c0", "c1", "c2", "c3", "c4"], groups)
        pipe.submit("base", 1,
                    np.array([[1.0, 1.0, 1.0, 0.0, 1.0],
                              [2.0, 2.0, 2.0, 1.0, 0.0],
                              [3.0, 3.0, 3.0, -1.0, 2.0]]),
                    np.ones((3, 5), dtype=bool), np.ones(3))
        first = pipe.publish()
        self.assertEqual(first["g"].decision, "published")
        self.assertEqual(first["h"].decision, "published")
        old_w = pipe.publisher.published("g").whitening.copy()
        old_means = pipe.publisher.published("g").means.copy()
        # 新片段让 g 的配对协方差严重不定，修正量超出上限；h 继续正常更新
        pipe.submit("conflict", 1,
                    np.array([[0.0, 1.0, -1.0, 0.0, 0.0],
                              [0.0, 2.0, -2.0, 0.0, 0.0],
                              [0.0, 3.0, -3.0, 0.0, 0.0]]),
                    np.array([[0, 1, 1, 0, 0]] * 3, dtype=bool),
                    np.array([1000.0, 1000.0, 1000.0]))
        pipe.submit("more-h", 1,
                    np.array([[0.0, 0.0, 0.0, 2.0, 2.0],
                              [0.0, 0.0, 0.0, 3.0, 1.0]]),
                    np.array([[0, 0, 0, 1, 1]] * 2, dtype=bool), np.ones(2))
        second = pipe.publish()
        self.assertEqual(second["g"].decision, "held")
        self.assertEqual(second["g"].version, 1)
        self.assertEqual(second["g"].block["reason"], "limits")
        self.assertGreater(second["g"].block["detail"]["adjust_fro_required"], 0.5)
        kept = pipe.publisher.published("g")
        self.assertEqual(kept.version, 1)
        np.testing.assert_array_equal(kept.whitening, old_w)
        np.testing.assert_array_equal(kept.means, old_means)  # 均值也保持旧版
        self.assertEqual(second["h"].decision, "published")
        self.assertEqual(second["h"].version, 2)
        # 部署侧：g 仍按旧版服务，h 用新版；同组内不混版本
        out = pipe.transform([0.5, 0.5, 0.5, 1.0, 1.0])
        self.assertEqual(out["served"]["g"]["version"], 1)
        self.assertEqual(out["served"]["h"]["version"], 2)
        y = np.asarray(out["served"]["g"]["values"])
        np.testing.assert_allclose(y, old_w @ (np.array([0.5, 0.5, 0.5]) - old_means),
                                   atol=1e-12)

    def test_no_previous_version_marks_unavailable(self):
        pipe = self.indefinite_pipeline(max_adjust_fro=0.01)
        decision = pipe.publish()["g"]
        self.assertEqual(decision.decision, "unavailable")
        self.assertIsNone(decision.version)
        self.assertEqual(decision.block["reason"], "limits")
        self.assertIn("g", pipe.transform([1.0, 2.0, 3.0])["skipped"])
        report = pipe.report()
        self.assertFalse(report["groups"]["g"]["available"])
        self.assertEqual(report["groups"]["g"]["block"]["reason"], "limits")


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.records, self.tensors = load_case_data()

    def test_partial_reading_served_per_group(self):
        result = run_case(self.records, self.tensors)
        served = result["transform"]["served"]
        skipped = result["transform"]["skipped"]
        # 读数缺 north：axis 组从未发布过，tool 组齐全
        self.assertIn("axis", skipped)
        self.assertIn("tool", served)
        self.assertEqual(served["tool"]["version"], 1)
        self.assertEqual(served["tool"]["channels"], ["axial", "radial"])
        # 用朴素统计独立重算 tool 组的白化结果
        rows = make_rows(self.tensors, ["a", "c"])
        cov = np.array([[naive_pair(rows, i, j)["cov"] for j in (3, 4)] for i in (3, 4)])
        evals, evecs = np.linalg.eigh(cov)
        w = (evecs * 1.0 / np.sqrt(np.maximum(evals, 1.0 / 36.0))) @ evecs.T
        w = (w + w.T) / 2
        means = np.array([naive_channel_mean(rows, 3), naive_channel_mean(rows, 4)])
        want = w @ (np.array([2.5, 1.2]) - means)
        np.testing.assert_allclose(served["tool"]["values"], want, atol=1e-9)

    def test_report_exposes_pairs_versions_and_reasons(self):
        result = run_case(self.records, self.tensors)
        report = result["report"]
        rows = make_rows(self.tensors, ["a", "c"])
        for gid, table in report["pairs"].items():
            for key, value in table.items():
                i, j = (self.records["channels"].index(n) for n in key.split("~"))
                want = naive_pair(rows, i, j)
                self.assertAlmostEqual(value["n_eff"], want["n_eff"], places=9)
                self.assertEqual(value["n"], want["n"])
        axis = report["groups"]["axis"]
        self.assertIsNone(axis["version"])
        self.assertFalse(axis["consumable"])
        self.assertEqual(axis["block"]["reason"], "limits")
        tool = report["groups"]["tool"]
        self.assertEqual(tool["version"], 1)
        self.assertTrue(tool["consumable"])
        # 发布决定与报告一致
        self.assertEqual(result["publish"]["axis"]["decision"], "unavailable")
        self.assertEqual(result["publish"]["tool"]["decision"], "published")

    def test_replay_is_deterministic(self):
        first = run_case(self.records, self.tensors)
        second = run_case(self.records, self.tensors)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
