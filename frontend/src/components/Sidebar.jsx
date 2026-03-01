import { api } from '../api';

export default function Sidebar({ settings, setSettings, indexStatus, onSave, onLoad }) {
    const { provider, topK, useReranker, temperature } = settings;

    const isIndexed = indexStatus?.is_indexed;

    return (
        <aside className="sidebar">
            <div className="sidebar-logo">
                <h1>📚 Advanced RAG</h1>
                <p>Research-grade multi-document Q&amp;A</p>
            </div>

            {/* Status */}
            <div className="sidebar-section">
                <div className="sidebar-section-title">Status</div>
                <div style={{ fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span className={`status-dot ${isIndexed ? 'green' : 'red'}`} />
                    {isIndexed
                        ? `${indexStatus.num_chunks ?? '?'} chunks indexed`
                        : 'No documents indexed'}
                </div>
            </div>

            {/* LLM Provider */}
            <div className="sidebar-section">
                <div className="sidebar-section-title">LLM</div>

                <div className="sidebar-field">
                    <label>Provider</label>
                    <select
                        value={provider}
                        onChange={e => setSettings(s => ({ ...s, provider: e.target.value }))}
                    >
                        <option value="groq">Groq — LLaMA-3.3-70B</option>
                        <option value="openai">OpenAI — GPT-4o-mini</option>
                        <option value="anthropic">Anthropic — Claude-3.5-Haiku</option>
                        <option value="together">Together.ai — Mistral-7B</option>
                    </select>
                </div>
            </div>

            {/* Retrieval */}
            <div className="sidebar-section">
                <div className="sidebar-section-title">Retrieval</div>

                <div className="sidebar-field">
                    <label>Top-K chunks: <span style={{ color: 'var(--accent-light)', fontWeight: 600 }}>{topK}</span></label>
                    <input
                        type="range" min={1} max={20} step={1}
                        value={topK}
                        onChange={e => setSettings(s => ({ ...s, topK: Number(e.target.value) }))}
                    />
                    <div className="range-row"><span>1</span><span>20</span></div>
                </div>

                <div className="toggle-row">
                    <label style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Cross-encoder reranker</label>
                    <button
                        className={`toggle ${useReranker ? 'on' : ''}`}
                        onClick={() => setSettings(s => ({ ...s, useReranker: !s.useReranker }))}
                        aria-label="Toggle reranker"
                    />
                </div>
            </div>

            {/* Generation */}
            <div className="sidebar-section">
                <div className="sidebar-section-title">Generation</div>

                <div className="sidebar-field">
                    <label>Temperature: <span style={{ color: 'var(--accent-light)', fontWeight: 600 }}>{temperature.toFixed(2)}</span></label>
                    <input
                        type="range" min={0} max={1} step={0.01}
                        value={temperature}
                        onChange={e => setSettings(s => ({ ...s, temperature: Number(e.target.value) }))}
                    />
                    <div className="range-row"><span>0.00</span><span>1.00</span></div>
                </div>
            </div>

            {/* Index Persistence */}
            <div className="sidebar-section">
                <div className="sidebar-section-title">Persistence</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <button
                        className="btn btn-secondary btn-sm btn-full"
                        onClick={onSave}
                        disabled={!isIndexed}
                    >
                        💾 Save Index
                    </button>
                    <button
                        className="btn btn-secondary btn-sm btn-full"
                        onClick={onLoad}
                    >
                        📂 Load Index
                    </button>
                </div>
            </div>

            {/* Stats footer */}
            {indexStatus?.embedding_model && (
                <div style={{ padding: '12px 16px', marginTop: 'auto', borderTop: '1px solid var(--border)' }}>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.8 }}>
                        <div>Model: <span style={{ color: 'var(--text-secondary)' }}>{indexStatus.embedding_model}</span></div>
                        {indexStatus.num_files != null && (
                            <div>Files: <span style={{ color: 'var(--text-secondary)' }}>{indexStatus.num_files}</span></div>
                        )}
                    </div>
                </div>
            )}
        </aside>
    );
}
