// GoodCredit — Credit Risk Scorecard Frontend

const form       = document.getElementById("predict-form");
const resultCard = document.getElementById("result-card");
const errorCard  = document.getElementById("error-card");
const submitBtn  = document.getElementById("submit-btn");
const btnText    = document.getElementById("btn-text");

const TIER_COLORS = {
  "Low":       "#22c55e",
  "Medium":    "#f59e0b",
  "High":      "#f97316",
  "Very High": "#ef4444",
};

function pct(v) { return v.toFixed(2) + "%"; }

function showResult(data) {
  // Icon & tier
  document.getElementById("result-icon").textContent = data.risk_icon;
  const tierEl = document.getElementById("result-tier");
  tierEl.textContent = data.risk_tier + " Risk";
  tierEl.style.color = data.risk_color;

  // Probabilities
  document.getElementById("res-bad-pct").textContent  = pct(data.bad_pct);
  document.getElementById("res-good-pct").textContent = pct(data.prob_good * 100);

  // Credit score
  document.getElementById("res-score").textContent = data.credit_score;

  // Risk bar (slider position %)
  const barPct = Math.min(data.bad_pct * 3, 98); // scale 0-33%+ across bar
  document.getElementById("risk-bar").style.left = barPct + "%";

  // Footer
  document.getElementById("result-footer").textContent =
    `Model Gini: ${data.model_gini}  ·  Benchmark: ${data.benchmark_gini}  ·  Raw P(Bad): ${data.prob_bad.toFixed(5)}`;

  // Header border accent
  document.querySelector(".result-header").style.borderBottomColor = data.risk_color + "55";

  resultCard.classList.remove("hidden");
  errorCard.classList.add("hidden");
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

function showError(msg) {
  document.getElementById("error-msg").textContent = "Error: " + msg;
  errorCard.classList.remove("hidden");
  resultCard.classList.add("hidden");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const payload = {};
  for (const el of form.elements) {
    if (el.name) {
      const v = parseFloat(el.value);
      payload[el.name] = isNaN(v) ? el.value : v;
    }
  }

  submitBtn.disabled = true;
  btnText.textContent = "⏳ Assessing …";
  resultCard.classList.add("hidden");
  errorCard.classList.add("hidden");

  try {
    const res  = await fetch("/api/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.success) {
      showResult(data);
    } else {
      showError(data.error || "Unknown error");
    }
  } catch (err) {
    showError("Network error: " + err.message);
  } finally {
    submitBtn.disabled = false;
    btnText.textContent = "🔍 Assess Credit Risk";
  }
});
