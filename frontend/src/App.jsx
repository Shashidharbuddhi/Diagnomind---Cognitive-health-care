import React, { useState } from "react";

export default function App() {
  const [notes, setNotes] = useState("");
  const [vitals, setVitals] = useState("");
  const [resp, setResp] = useState(null);
  const [loading, setLoading] = useState(false);

  async function submit() {
    setLoading(true);
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
      const r = await fetch("http://localhost:8000/query?top_k=5", {
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
    <div style={{ padding: 20, fontFamily: "Arial" }}>
      <h2>DiagnoMind Prototype</h2>
      <textarea
        rows={8}
        cols={80}
        placeholder="Patient notes..."
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
      />
      <div style={{ marginTop: 10 }}>
        <input
          style={{ width: 500 }}
          placeholder="vitals: heart_rate=110,bp_systolic=90,spo2=88"
          value={vitals}
          onChange={(e) => setVitals(e.target.value)}
        />
      </div>
      <button style={{ marginTop: 10 }} onClick={submit} disabled={loading}>
        {loading ? "Querying..." : "Query"}
      </button>
      {resp && (
        <pre style={{ marginTop: 20, whiteSpace: "pre-wrap" }}>
          {JSON.stringify(resp, null, 2)}
        </pre>
      )}
    </div>
  );
}
