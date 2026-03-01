import { useState, useEffect } from 'react';
import { api } from '../api';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, Legend
} from 'recharts';

const COLORS = {
    faithfulness: '#60a5fa',
    confidence: '#34d399',
    latency: '#a78bfa',
};

function MetricCard({ label, value, sub, color }) {
    return (
        <div className="metric-card">
            <div className="metric-card-title">{label}</div>
            <div className="metric-card-value" style={{ color: color || 'var(--accent-light)' }}>
                {value ?? '—'}
            </div>
            {sub && <div className="metric-card-sub">{sub}</div>}
        </div>
    );
}

const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    return (
        <div style={{
            background: 'var(--bg-surface)', border: '1px solid var(--border)',
            borderRadius: 8, padding: '10px 14px', fontSize: 12,
        }}>
            {payload.map(p => (
                <div key={p.dataKey} style={{ color: p.color }}>
                    {p.name}: {typeof p.value === 'number' ? p.value.toFixed(3) : p.value}
                </div>
            ))}
        </div>
    );
};

export default function EvaluationTab() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    async function load() {
        setLoading(true);
        try {
            const res = await api.evaluation();
            setData(res);
            setError(null);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }

    useEffect(() => { load(); }, []);

    if (loading) return <div style={{ color: 'var(--text-muted)', padding: 40, textAlign: 'center' }}>Loading metrics…</div>;
    if (error) return <div className="alert alert-error">Error: {error}</div>;

    const { summary = {}, query_records = [] } = data || {};

    const chartData = query_records.slice(-30).map((r, i) => ({
        idx: i + 1,
        faithfulness: r.faithfulness ?? null,
        confidence: r.confidence ?? null,
        latency: r.latency_ms ? +(r.latency_ms / 1000).toFixed(2) : null,
    }));

    return (
        <div style={{ maxWidth: 1000, margin: '0 auto' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
                <div>
                    <h2 style={{ fontSize: 20, fontWeight: 700 }}>📊 Evaluation Dashboard</h2>
                    <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>Metrics across all queries in this session</p>
                </div>
                <button className="btn btn-secondary btn-sm" onClick={load}>↻ Refresh</button>
            </div>

            {/* Metric cards */}
            <div className="eval-grid">
                <MetricCard
                    label="Mean Faithfulness"
                    value={summary.mean_faithfulness != null ? (summary.mean_faithfulness * 100).toFixed(1) + '%' : null}
                    sub="Groundedness in retrieved context"
                    color={summary.mean_faithfulness > 0.7 ? 'var(--green)' : summary.mean_faithfulness > 0.4 ? 'var(--yellow)' : 'var(--red)'}
                />
                <MetricCard
                    label="Mean Confidence"
                    value={summary.mean_confidence != null ? (summary.mean_confidence * 100).toFixed(1) + '%' : null}
                    sub="Average retrieval chunk score"
                    color="var(--accent-light)"
                />
                <MetricCard
                    label="Mean Latency"
                    value={summary.mean_latency_ms != null ? (summary.mean_latency_ms / 1000).toFixed(2) + 's' : null}
                    sub="End-to-end query latency"
                    color="var(--purple)"
                />
                <MetricCard
                    label="Total Queries"
                    value={summary.total_queries ?? 0}
                    sub="Logged in eval_log.jsonl"
                    color="var(--text-primary)"
                />
            </div>

            {/* Chart */}
            {chartData.length > 0 && (
                <div className="chart-wrap">
                    <div className="chart-title">Query History (last 30)</div>
                    <ResponsiveContainer width="100%" height={240}>
                        <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                            <CartesianGrid stroke="rgba(255,255,255,0.04)" strokeDasharray="3 3" />
                            <XAxis dataKey="idx" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                            <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} domain={[0, 1]} />
                            <Tooltip content={<CustomTooltip />} />
                            <Legend wrapperStyle={{ fontSize: 12 }} />
                            <Line type="monotone" dataKey="faithfulness" name="Faithfulness" stroke={COLORS.faithfulness} strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                            <Line type="monotone" dataKey="confidence" name="Confidence" stroke={COLORS.confidence} strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            )}

            {/* Query log table */}
            {query_records.length > 0 && (
                <div className="query-log">
                    <div className="query-log-title">Recent Queries</div>
                    <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
                        <div className="query-row header">
                            <div>Question</div>
                            <div>Faithfulness</div>
                            <div>Confidence</div>
                            <div>Latency</div>
                            <div>Provider</div>
                        </div>
                        {[...query_records].reverse().slice(0, 20).map((r, i) => (
                            <div key={i} className="query-row">
                                <div className="question" title={r.question}>{r.question}</div>
                                <div>
                                    <span className={`badge ${r.faithfulness > 0.7 ? 'badge-green' : r.faithfulness > 0.4 ? 'badge-yellow' : 'badge-red'}`}>
                                        {r.faithfulness != null ? (r.faithfulness * 100).toFixed(0) + '%' : '—'}
                                    </span>
                                </div>
                                <div>
                                    <span className="badge badge-blue">
                                        {r.confidence != null ? (r.confidence * 100).toFixed(0) + '%' : '—'}
                                    </span>
                                </div>
                                <div style={{ color: 'var(--text-muted)' }}>
                                    {r.latency_ms != null ? `${(r.latency_ms / 1000).toFixed(2)}s` : '—'}
                                </div>
                                <div style={{ color: 'var(--text-muted)' }}>{r.provider || '—'}</div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {query_records.length === 0 && (
                <div style={{ textAlign: 'center', padding: '60px 0', color: 'var(--text-muted)' }}>
                    <div style={{ fontSize: 32, marginBottom: 12 }}>📭</div>
                    <p>No query data yet. Ask some questions in the Chat tab first.</p>
                </div>
            )}
        </div>
    );
}
