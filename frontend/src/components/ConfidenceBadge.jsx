import React from 'react';

export default function ConfidenceBadge({ score, breakdown }) {
  const pct = Math.round((score || 0) * 100);
  const color = pct >= 80 ? '#10b981' : pct >= 60 ? '#f59e0b' : '#ef4444';

  return (
    <div className="confidence-meter" title={JSON.stringify(breakdown || {})}>
      <div className="confidence-bar-bg">
        <div
          className="confidence-bar-fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className="confidence-text" style={{ color }}>
        {pct}% Conf
      </span>
    </div>
  );
}
