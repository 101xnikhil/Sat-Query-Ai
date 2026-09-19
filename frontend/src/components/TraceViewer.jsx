import React, { useState } from 'react';
import { Terminal, CheckCircle2, AlertTriangle, Clock, ChevronDown, ChevronUp, Code } from 'lucide-react';

export default function TraceViewer({ trace }) {
  const [isOpen, setIsOpen] = useState(true);
  const [showRawJson, setShowRawJson] = useState(false);

  if (!trace) return null;

  const val = trace.input_validation || {};
  const tools = trace.tool_calls || [];

  return (
    <div className="trace-card">
      <div className="trace-header-toggle" onClick={() => setIsOpen(!isOpen)}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Terminal size={15} color="var(--accent-cyan)" />
          <span>OBSERVABLE EXECUTION TRACE</span>
          <span style={{ fontSize: '0.7rem', color: 'var(--accent-emerald)', fontFamily: 'var(--font-mono)' }}>
            ({trace.total_duration_ms}ms)
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowRawJson(!showRawJson);
            }}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--accent-cyan)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              fontSize: '0.72rem'
            }}
          >
            <Code size={13} />
            {showRawJson ? 'Formatted' : 'JSON'}
          </button>
          {isOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
      </div>

      {isOpen && (
        <div className="trace-content">
          {showRawJson ? (
            <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
              {JSON.stringify(trace, null, 2)}
            </pre>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {/* Validation Step */}
              <div style={{ borderLeft: '2px solid #10b981', paddingLeft: 8 }}>
                <div style={{ fontWeight: 600, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: 6 }}>
                  <CheckCircle2 size={13} color="#10b981" />
                  <span>Input Validation & Harmonization</span>
                </div>
                <div style={{ color: 'var(--text-secondary)', marginTop: 2, fontSize: '0.7rem' }}>
                  Passed: {String(val.passed)} | Overlap IoU: {val.overlap_iou !== null ? val.overlap_iou : 'N/A'}
                  {val.sar_preprocessing && val.sar_preprocessing.length > 0 && (
                    <span> | SAR Preprocessing: dB + Lee Filter</span>
                  )}
                  {val.reprojections && val.reprojections.length > 0 && (
                    <span> | Auto-Reprojected: {val.reprojections.length} raster(s)</span>
                  )}
                </div>
              </div>

              {/* Tool Execution Sequence */}
              <div>
                <div style={{ fontWeight: 600, color: '#94a3b8', marginBottom: 4, fontSize: '0.7rem', textTransform: 'uppercase' }}>
                  Ordered Tool Calls ({tools.length})
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {tools.map((t, idx) => (
                    <div
                      key={idx}
                      style={{
                        background: 'rgba(255,255,255,0.03)',
                        padding: '6px 8px',
                        borderRadius: '6px',
                        border: '1px solid rgba(255,255,255,0.05)',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>
                          {idx + 1}. {t.tool_name}
                        </span>
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.68rem', display: 'flex', alignItems: 'center', gap: 3 }}>
                          <Clock size={11} /> {t.duration_ms}ms
                        </span>
                      </div>
                      <div style={{ color: 'var(--text-muted)', fontSize: '0.68rem', marginTop: 2 }}>
                        Whitelisted Params: {JSON.stringify(t.parameters)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Observable Output Metrics */}
              {trace.computed_metrics && Object.keys(trace.computed_metrics).length > 0 && (
                <div style={{ borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: 6 }}>
                  <span style={{ color: '#94a3b8', fontSize: '0.7rem', textTransform: 'uppercase' }}>
                    Observable Mask Metrics:
                  </span>
                  <div style={{ color: 'var(--accent-emerald)', fontSize: '0.7rem', marginTop: 2 }}>
                    {JSON.stringify(trace.computed_metrics)}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
