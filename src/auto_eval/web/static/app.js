import { createApp, ref, computed, onMounted } from "https://unpkg.com/vue@3/dist/vue.esm-browser.js";

createApp({
  setup() {
    const modes = [
      { key: "single", label: "单回答盲评" },
      { key: "compare", label: "两回答对比" },
      { key: "online", label: "接模型在线评估" },
      { key: "process", label: "过程盲评(含轨迹)" },
    ];
    const mode = ref("single");
    const text = ref("");
    const fileText = ref("");
    const isJsonl = ref(false);
    const items = ref([]);
    const errors = ref([]);
    const judges = ref([]);
    const models = ref([]);
    const selectedJudges = ref([]);
    const selectedModel = ref("");
    const concurrency = ref(4);
    const running = ref(false);
    const progress = ref(0);
    const total = ref(0);
    const results = ref([]);
    const summary = ref(null);
    const taskId = ref("");

    const formatHint = computed(
      () =>
        ({
          single: "每行一题：query ||| answer [||| reference]   （reference 可选，填了会跑元评测校验裁判）",
          compare: "每行一题：query ||| answerA ||| answerB [||| reference]",
          online: "每行一题：query [||| reference]   （后端现场调模型生成回答，再盲评）",
          process: "每行一题：query ||| answer ||| trace [||| reference]   （trace=被测 agent 的推理/工具轨迹；评过程质量）",
        }[mode.value])
    );
    const placeholder = computed(
      () =>
        ({
          single: "光合作用是什么？ ||| 植物利用光合成有机物 ||| 绿色植物利用光能将二氧化碳和水合成有机物并释放氧气\n中国最长的河流？ ||| 长江",
          compare: "写一首关于春天的诗 ||| 春风又绿江南岸 ||| 春眠不觉晓\n推荐一部科幻电影 ||| 星际穿越 ||| 流浪地球",
          online: "2024 年诺贝尔文学奖获得者是谁？\n计算 17 × 24 等于多少？",
          process: "北京到上海多少公里？ ||| 约 1200 公里 ||| 1.调用地图API 2.距离=1318km 3.约1200km\n某函数是否正确？ ||| 正确 ||| def f(n): return 1 if n<=1 else n*f(n-1)",
        }[mode.value])
    );

    const previewKeys = computed(() => {
      if (!items.value.length) return [];
      const keys = ["query"];
      if (mode.value === "single") keys.push("answer", "reference");
      else if (mode.value === "compare") keys.push("answer_a", "answer_b", "reference");
      else if (mode.value === "process") keys.push("answer", "trace", "reference");
      else keys.push("reference");
      return keys.filter((k) => items.value.some((it) => it[k] != null && it[k] !== ""));
    });

    const resultCols = computed(() => {
      if (mode.value === "compare")
        return [
          { key: "query", label: "题目" },
          { key: "answer_a", label: "回答 A" },
          { key: "answer_b", label: "回答 B" },
          { key: "winner", label: "胜者" },
          { key: "bidirectional_consistent", label: "双向一致" },
          { key: "rationale", label: "理由" },
        ];
      return [
        { key: "query", label: "题目" },
        { key: mode.value === "online" ? "generated_answer" : "answer", label: mode.value === "online" ? "生成回答" : "回答" },
        { key: "correctness", label: "判定" },
        { key: "total", label: "总分" },
        { key: "rubric", label: "各维度" },
        { key: "used_search", label: "联网" },
        { key: "truncated", label: "截断" },
        { key: "arbitrated", label: "仲裁" },
        { key: "agree", label: "与真值" },
        { key: "rationale", label: "理由" },
      ];
    });

    function trunc(v) {
      if (v == null) return "";
      const s = String(v);
      return s.length > 50 ? s.slice(0, 50) + "…" : s;
    }

    function switchMode(k) {
      mode.value = k;
      items.value = [];
      errors.value = [];
      fileText.value = "";
      isJsonl.value = false;
    }

    function onFile(e) {
      const f = e.target.files[0];
      if (!f) return;
      const r = new FileReader();
      r.onload = () => {
        fileText.value = r.result;
        text.value = r.result;
        isJsonl.value = true;
      };
      r.readAsText(f, "utf-8");
    }

    async function doParse() {
      const body = { mode: mode.value };
      if (isJsonl.value && fileText.value) body.jsonl = fileText.value;
      else body.text = text.value;
      const r = await fetch("/api/parse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const d = await r.json();
      items.value = d.items;
      errors.value = d.errors;
      if (errors.value.length) console.log("解析错误：", errors.value);
    }

    async function submit() {
      // 自动解析最新输入（用户可跳过手动"解析预览"）
      await doParse();
      if (!items.value.length) {
        alert("解析后没有可评估的题。请检查格式：每行『问题 ||| 回答』。");
        return;
      }
      results.value = [];
      summary.value = null;
      progress.value = 0;
      total.value = items.value.length;
      running.value = true;
      const body = {
        mode: mode.value,
        items: items.value,
        options: {
          judges: selectedJudges.value,
          model: selectedModel.value,
          concurrency: concurrency.value,
        },
      };
      const r = await fetch("/api/eval", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const d = await r.json();
      taskId.value = d.task_id;
      connectSSE();
    }

    function connectSSE() {
      const es = new EventSource(`/api/eval/${taskId.value}/stream`);
      es.addEventListener("result", (e) => {
        const d = JSON.parse(e.data);
        results.value.push(d.result);
        progress.value = d.progress;
      });
      es.addEventListener("done", (e) => {
        summary.value = JSON.parse(e.data).summary;
        running.value = false;
        es.close();
      });
      es.addEventListener("error", (e) => {
        try {
          if (e.data) {
            const d = JSON.parse(e.data);
            alert("评估出错：" + (d.message || "未知"));
          }
        } catch (_) {}
        running.value = false;
        es.close();
      });
    }

    function cell(r, c) {
      const v = r[c.key];
      if (c.key === "rubric" && v) return Object.entries(v).map(([k, x]) => `${k}:${x}`).join("  ");
      if (c.key === "agree") {
        if (v === undefined) return "";
        return v === true ? "✓ 一致" : v === false ? "✗ 不一致" : "?";
      }
      if (c.key === "used_search") return v ? "是" : "否";
      if (c.key === "truncated") return v ? "⚠️是(强制判定)" : "";
      if (c.key === "arbitrated") return v ? `⚖️是(${r.arbitrator_confidence ?? "-"})` : "";
      if (c.key === "bidirectional_consistent") return v ? "是" : "否(位置偏差)";
      if (c.key === "winner") return v === "a" ? "A" : v === "b" ? "B" : "平";
      if (c.key === "correctness") return ({ right: "正确", wrong: "错误", partial: "部分", unclear: "不清" }[v] || v) || "";
      if (v == null) return "";
      if (typeof v === "string" && v.length > 80) return v.slice(0, 80) + "…";
      return v;
    }

    function exportCsv() {
      window.open(`/api/eval/${taskId.value}/export?format=csv`);
    }
    function exportJson() {
      window.open(`/api/eval/${taskId.value}/export?format=json`);
    }

    onMounted(async () => {
      const r = await fetch("/api/config");
      const d = await r.json();
      judges.value = d.judges;
      models.value = d.models;
      selectedJudges.value = d.judges.length ? [d.judges[0].name] : [];
      selectedModel.value = d.models[0] || "";
    });

    return {
      modes, mode, text, items, errors, judges, models, selectedJudges, selectedModel,
      concurrency, running, progress, total, results, summary, taskId,
      formatHint, placeholder, previewKeys, resultCols,
      trunc, switchMode, onFile, doParse, submit, cell, exportCsv, exportJson,
    };
  },
}).mount("#app");
