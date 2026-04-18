import React, { useState } from "react";
import "./App.css";

export default function App() {
  const [notes, setNotes] = useState("");
  const [vitals, setVitals] = useState("");
  const [resp, setResp] = useState(null);
  const [loading, setLoading] = useState(false);
  const [topK, setTopK] = useState(5);

  const hasResponse = !!resp;
  const hasError = !!resp?.error || !!resp?.detail;
  const retrievedCases = Array.isArray(resp?.retrieved_cases) ? resp.retrieved_cases : [];
  const distances = Array.isArray(resp?.distances) ? resp.distances : [];

  async function submit() {
    setLoading(true);
    setResp(null);
    let vitObj = {};
    if (vitals.trim()) {
      vitals.split(",").forEach((pair) => {
        const [k, v] = pair.split("=").map((s) => s.trim());
        if (k) vitObj[k] = isNaN(v) ? v : Number(v);
      });
    }
    const payload = {
      subject_id: 0,
      hadm_id: 0,
      vitals: vitObj,
      notes
    };
    try {
      const r = await fetch(`http://localhost:8000/query?top_k=${topK}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const j = await r.json();
      setResp(j);
    } catch (e) {
      setResp({ error: String(e) });
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />

      <section className="hero card">
        <p className="eyebrow">Clinical Decision Support</p>
        <h1>DiagnoMind</h1>
        <p className="subtext">
          Enter patient notes and vitals to retrieve similar cases and generate reasoning.
        </p>
      </section>

      <section className="card form-card">
        <div className="field-group">
          <label htmlFor="notes">Patient Notes</label>
          <textarea
            id="notes"
            className="notes"
            placeholder="Patient presents with shortness of breath, chest tightness..."
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
          />
        </div>

        <div className="inline-fields">
          <div className="field-group">
            <label htmlFor="vitals">Vitals</label>
            <input
              id="vitals"
              className="vitals"
              placeholder="heart_rate=110,bp_systolic=90,spo2=88"
              value={vitals}
              onChange={(e) => setVitals(e.target.value)}
            />
          </div>

          <div className="field-group narrow">
            <label htmlFor="topk">Top K</label>
            <input
              id="topk"
              className="topk"
              type="number"
              min={1}
              max={10}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value) || 5)}
            />
          </div>
        </div>

        <div className="actions">
          <button className="primary-btn" onClick={submit} disabled={loading || !notes.trim()}>
            {loading ? "Analyzing..." : "Run Analysis"}
          </button>
          <span className="hint">Backend: http://localhost:8000</span>
        </div>
      </section>

      {hasResponse && (
        <section className="results-grid">
          <article className="card status-card">
            <h2>Request Status</h2>
            <p className={`status-pill ${hasError ? "error" : "ok"}`}>
              {hasError ? "Error" : resp?.status || "Success"}
            </p>
            {hasError && (
              <pre className="error-block">
                {resp?.detail || resp?.error || "Unknown error"}
              </pre>
            )}
            {!hasError && (
              <p className="meta">
                Retrieved <strong>{resp?.retrieved_count ?? 0}</strong> cases
              </p>
            )}
          </article>

          {!hasError && (
            <>
              <article className="card">
                <h2>Patient Summary</h2>
                <pre className="text-block">{resp?.patient_summary || "No summary available."}</pre>
              </article>

              <article className="card">
                <h2>Clinical Reasoning</h2>
                <pre className="text-block">{resp?.reasoning || "No reasoning generated."}</pre>
              </article>

              <article className="card span-2">
                <h2>Retrieved Cases</h2>
                {retrievedCases.length === 0 ? (
                  <p className="meta">No similar cases returned.</p>
                ) : (
                  <div className="cases-list">
                    {retrievedCases.map((item, index) => (
                      <div className="case-item" key={`${index}-${distances[index] ?? "na"}`}>
                        <div className="case-head">
                          <strong>Case {index + 1}</strong>
                          {typeof distances[index] === "number" && (
                            <span>L2 distance: {distances[index].toFixed(3)}</span>
                          )}
                        </div>
                        <pre className="text-block">{item}</pre>
                      </div>
                    ))}
                  </div>
                )}
              </article>
            </>
          )}
        </section>
      )}

      {!hasResponse && !loading && (
        <section className="card empty-state">
          <h2>Ready for Analysis</h2>
          <p className="meta">
            Add notes, optional vitals, and click <strong>Run Analysis</strong>.
          </p>
        </section>
      )}

      {loading && (
        <section className="card loading-state">
          <div className="pulse" />
          <p className="meta">Querying the backend and generating reasoning...</p>
        </section>
      )}
    </main>
  );
}
