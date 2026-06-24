from __future__ import annotations

import streamlit as st


def inject_styles() -> None:
    st.markdown(
        """
<style>
.wc-hero {
  padding: 1rem 1.1rem;
  border: 1px solid #d8dee8;
  border-radius: 8px;
  background: #f8fafc;
  color: #172033;
  margin-bottom: 1rem;
}
.wc-hero strong {
  color: #111827;
}
.match-card {
  border: 1px solid #d5dce7;
  border-radius: 8px;
  padding: 1rem;
  margin: .75rem 0;
  background: #ffffff;
  box-shadow: 0 1px 2px rgba(15, 23, 42, .06);
}
.match-head {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: center;
  margin-bottom: .75rem;
}
.teams {
  font-size: 1.15rem;
  font-weight: 700;
  color: #172033;
}
.status-badge {
  display: inline-block;
  border-radius: 999px;
  padding: .18rem .55rem;
  font-size: .78rem;
  font-weight: 700;
  border: 1px solid #c7d2fe;
  color: #27346a;
  background: #eef2ff;
  white-space: nowrap;
}
.live-badge {
  border-color: #fecaca;
  color: #8a1c1c;
  background: #fff1f2;
}
.prob-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: .55rem;
  margin-top: .75rem;
}
.prob-chip {
  border: 1px solid #d6dee9;
  border-radius: 8px;
  padding: .65rem .55rem;
  background: #f9fbfd;
  text-align: center;
}
.prob-chip strong {
  display: block;
  font-size: 1.42rem;
  color: #111827;
}
.prob-chip span {
  color: #475569;
  font-size: .82rem;
}
.prob-chip.best {
  border-color: #9cc7b4;
  background: #edf8f2;
}
.factor-row {
  color: #475569;
  font-size: .86rem;
  margin-top: .5rem;
}
.muted-note {
  color: #64748b;
  font-size: .88rem;
}
@media (max-width: 760px) {
  .match-head { display: block; }
  .prob-grid { grid-template-columns: 1fr; }
}
</style>
""",
        unsafe_allow_html=True,
    )
