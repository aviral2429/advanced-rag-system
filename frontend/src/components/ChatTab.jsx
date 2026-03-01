import { useState, useRef, useEffect } from 'react';
import { api } from '../api';

function ConfidenceBar({ value }) {
    const pct = Math.round((value || 0) * 100);
    const color = value > 0.7 ? 'var(--green)' : value > 0.4 ? 'var(--yellow)' : 'var(--red)';
    return (
        <div className="confidence-bar-wrap">
            <div className="confidence-bar">
                <div className="confidence-bar-fill" style={{ width: `${pct}%`, background: color }} />
            </div>
            <span>{pct}%</span>
        </div>
    );
}

function Message({ msg }) {
    const isUser = msg.role === 'user';

    return (
        <div className={`message ${isUser ? 'user' : 'assistant'}`}>
            <div className="message-avatar">
                {isUser ? 'U' : '🤖'}
            </div>
            <div className="message-body">
                {msg.thinking ? (
                    <div className="thinking-indicator">
                        <div className="dots">
                            <div className="dot" /><div className="dot" /><div className="dot" />
                        </div>
                        Thinking…
                    </div>
                ) : (
                    <>
                        <div className="message-bubble">
                            {msg.content}
                        </div>

                        {msg.hallucination_warning && (
                            <div className="hallucination-warning">
                                ⚠️ Possible hallucination — faithfulness score low
                            </div>
                        )}

                        {!isUser && msg.confidence != null && (
                            <div className="message-meta">
                                <span className="badge badge-blue">
                                    {msg.retrieval_method || 'hybrid'}
                                </span>
                                <ConfidenceBar value={msg.confidence} />
                                {msg.faithfulness != null && (
                                    <span className={`badge ${msg.faithfulness > 0.7 ? 'badge-green' : msg.faithfulness > 0.4 ? 'badge-yellow' : 'badge-red'}`}>
                                        Faithfulness {Math.round(msg.faithfulness * 100)}%
                                    </span>
                                )}
                                {msg.latency_ms != null && (
                                    <span className="badge badge-gray">{Math.round(msg.latency_ms)}ms</span>
                                )}
                            </div>
                        )}

                        {msg.sources?.length > 0 && (
                            <details style={{ marginTop: 4 }}>
                                <summary style={{ fontSize: 12, color: 'var(--text-muted)', cursor: 'pointer', marginBottom: 6 }}>
                                    📎 {msg.sources.length} sources
                                </summary>
                                <div className="sources-list">
                                    {msg.sources.map((src, i) => (
                                        <div key={i} className="source-chip">
                                            <strong>#{i + 1}</strong>&nbsp;
                                            {src.source || src.filename || 'Document'} — p.{src.page ?? '?'}
                                            {src.score != null && (
                                                <span style={{ marginLeft: 8, opacity: 0.6 }}>score: {src.score.toFixed(3)}</span>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </details>
                        )}
                    </>
                )}
            </div>
        </div>
    );
}

export default function ChatTab({ settings, isIndexed }) {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const bottomRef = useRef(null);
    const textareaRef = useRef(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    async function send() {
        const q = input.trim();
        if (!q || loading) return;

        setInput('');
        setError(null);

        setMessages(prev => [
            ...prev,
            { id: Date.now(), role: 'user', content: q },
            { id: Date.now() + 1, role: 'assistant', thinking: true },
        ]);
        setLoading(true);

        try {
            const res = await api.query(q, {
                top_k: settings.topK,
                use_reranker: settings.useReranker,
                temperature: settings.temperature,
            });

            setMessages(prev => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                    id: Date.now(),
                    role: 'assistant',
                    content: res.answer,
                    sources: res.sources,
                    confidence: res.confidence,
                    faithfulness: res.faithfulness,
                    hallucination_warning: res.hallucination_warning,
                    latency_ms: res.latency_ms,
                    retrieval_method: res.retrieval_method,
                };
                return updated;
            });
        } catch (err) {
            setError(err.message);
            setMessages(prev => prev.slice(0, -1)); // remove thinking bubble
        } finally {
            setLoading(false);
        }
    }

    function handleKey(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            send();
        }
    }

    return (
        <div className="chat-container">
            <div className="chat-messages">
                {messages.length === 0 && (
                    <div className="chat-empty">
                        <div className="chat-empty-icon">💬</div>
                        {isIndexed
                            ? <p>Ask a question about your documents</p>
                            : <p>Upload and index at least one PDF to start asking questions.</p>
                        }
                    </div>
                )}

                {messages.map(msg => (
                    <Message key={msg.id} msg={msg} />
                ))}

                {error && (
                    <div className="alert alert-error">⚠ Error: {error}</div>
                )}
                <div ref={bottomRef} />
            </div>

            <div className="chat-input-wrap">
                <div className="chat-input-box">
                    <textarea
                        ref={textareaRef}
                        rows={1}
                        value={input}
                        onChange={e => {
                            setInput(e.target.value);
                            e.target.style.height = 'auto';
                            e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px';
                        }}
                        onKeyDown={handleKey}
                        placeholder={isIndexed ? 'Ask a question about your documents…' : 'Index documents first…'}
                        disabled={loading || !isIndexed}
                    />
                    <button
                        className="send-btn"
                        onClick={send}
                        disabled={loading || !isIndexed || !input.trim()}
                        aria-label="Send"
                    >
                        ➤
                    </button>
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6, textAlign: 'right' }}>
                    Enter to send · Shift+Enter for new line
                </div>
            </div>
        </div>
    );
}
