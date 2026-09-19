import React, { useState } from 'react';
import { Send, MessageSquare, Download, FileText, Sparkles, AlertCircle } from 'lucide-react';
import ConfidenceBadge from './ConfidenceBadge';
import TraceViewer from './TraceViewer';

export default function ChatPanel({
  currentMode,
  uploadedImages,
  onRunQuery,
  queryResponse,
  loading,
  error
}) {
  const [prompt, setPrompt] = useState('');
  const [pdfDownloading, setPdfDownloading] = useState(false);

  const suggestions = {
    single: [
      "Where are the runways or aircraft located?",
      "Describe the land use and features of this scene",
      "Calculate NDVI vegetation index coverage",
      "What is the primary terrain or land cover present?"
    ],
    optical_sar: [
      "Identify built-up and water regions using both sensors",
      "Segment surface water bodies and compute area",
      "Classify urban structures using SAR backscatter"
    ],
    bitemporal: [
      "What changed between date T1 and date T2?",
      "Generate a change map and calculate total area changed",
      "Detect forest canopy loss and urban expansion"
    ]
  };

  const handleSend = (queryText) => {
    const text = queryText || prompt;
    if (!text.trim() || uploadedImages.length === 0) return;
    onRunQuery(text);
  };

  const handleDownloadPDF = async () => {
    if (!queryResponse) return;
    setPdfDownloading(true);
    try {
      const pdfUrl = queryResponse.pdf_report_url || `/api/reports/${queryResponse.session_id}/pdf`;
      const res = await fetch(pdfUrl);
      if (!res.ok) throw new Error('PDF generation failed on server.');
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `SatQuery_Report_${queryResponse.session_id}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('PDF download error:', err);
      alert('Could not download PDF report: ' + err.message);
    } finally {
      setPdfDownloading(false);
    }
  };

  const handleDownloadJSON = () => {
    if (!queryResponse) return;
    const reportData = {
      title: "SatQuery AI Remote Sensing Evidence Report",
      timestamp: new Date().toISOString(),
      query: queryResponse.query,
      task: queryResponse.task,
      answer: queryResponse.answer,
      confidence: queryResponse.confidence_score,
      computed_metrics: queryResponse.computed_metrics,
      trace: queryResponse.trace
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `SatQuery_Report_${queryResponse.session_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="panel right">
      <div className="panel-header">
        <span>INTELLIGENT QUERY & EVIDENCE</span>
        <MessageSquare size={16} color="var(--accent-cyan)" />
      </div>

      <div className="panel-body">
        {/* Quick Query Pills */}
        <div>
          <label className="overlay-title" style={{ display: 'block', marginBottom: 6 }}>
            Recommended Queries
          </label>
          <div className="quick-prompts">
            {(suggestions[currentMode] || []).map((s, idx) => (
              <button
                key={idx}
                className="prompt-chip"
                onClick={() => {
                  setPrompt(s);
                  handleSend(s);
                }}
              >
                {s}
              </button>
            ))}
          </div>
        </div>

        {/* Input Field */}
        <div className="query-box">
          <div className="input-group">
            <input
              type="text"
              className="query-input"
              placeholder={
                uploadedImages.length === 0
                  ? "Upload imagery first..."
                  : "Ask natural language question (e.g. Locate ships, Segment water...)"
              }
              value={prompt}
              disabled={uploadedImages.length === 0 || loading}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            />
            <button
              className="query-btn"
              disabled={uploadedImages.length === 0 || !prompt.trim() || loading}
              onClick={() => handleSend()}
            >
              <Send size={15} />
              <span>{loading ? 'Running...' : 'Query'}</span>
            </button>
          </div>
        </div>

        {error && (
          <div style={{ color: 'var(--accent-rose)', fontSize: '0.8rem', display: 'flex', gap: 6, alignItems: 'center' }}>
            <AlertCircle size={15} />
            <span>{error}</span>
          </div>
        )}

        {/* Grounded Answer Card */}
        {queryResponse && (
          <div className="answer-card">
            <div className="answer-header">
              <span className="task-badge">{queryResponse.task}</span>
              <ConfidenceBadge
                score={queryResponse.confidence_score}
                breakdown={queryResponse.trace?.confidence_breakdown}
              />
            </div>

            <div className="answer-body">
              {queryResponse.answer}
            </div>

            {/* Empirical Mask-Derived Metrics */}
            {queryResponse.computed_metrics && Object.keys(queryResponse.computed_metrics).length > 0 && (
              <div className="metrics-grid">
                {Object.entries(queryResponse.computed_metrics)
                  .filter(([k]) => !['labels', 'errors', 'validation_passed'].includes(k))
                  .map(([key, val]) => (
                    <div key={key} className="metric-box">
                      <span className="metric-label">{key.replace(/_/g, ' ')}</span>
                      <span className="metric-value">
                        {typeof val === 'number' ? val.toLocaleString() : String(val)}
                      </span>
                    </div>
                  ))}
              </div>
            )}

            {/* Download Evidence Report Buttons */}
            <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
              <button
                onClick={handleDownloadPDF}
                disabled={pdfDownloading}
                style={{
                  flex: 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 6,
                  background: 'rgba(0, 229, 255, 0.12)',
                  border: '1px solid var(--accent-cyan)',
                  color: 'var(--accent-cyan)',
                  padding: '9px 12px',
                  borderRadius: '8px',
                  fontSize: '0.78rem',
                  fontWeight: 600,
                  cursor: pdfDownloading ? 'wait' : 'pointer',
                  transition: 'all 0.2s',
                }}
              >
                <FileText size={14} />
                <span>{pdfDownloading ? 'Generating PDF...' : 'Download PDF Report'}</span>
              </button>
              <button
                onClick={handleDownloadJSON}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 5,
                  background: 'rgba(255, 255, 255, 0.05)',
                  border: '1px solid var(--border-color)',
                  color: 'var(--text-muted)',
                  padding: '9px 12px',
                  borderRadius: '8px',
                  fontSize: '0.78rem',
                  fontWeight: 500,
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
                title="Export raw JSON metadata and execution trace"
              >
                <Download size={13} />
                <span>JSON</span>
              </button>
            </div>
          </div>
        )}

        {/* Trace Inspector */}
        {queryResponse?.trace && (
          <TraceViewer trace={queryResponse.trace} />
        )}
      </div>
    </div>
  );
}
