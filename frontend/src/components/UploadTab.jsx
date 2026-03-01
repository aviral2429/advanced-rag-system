import { useState, useCallback } from 'react';
import { api } from '../api';

function formatBytes(bytes) {
    if (!bytes) return '0 B';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

export default function UploadTab({ onIndexed }) {
    const [files, setFiles] = useState([]);
    const [dragging, setDragging] = useState(false);
    const [status, setStatus] = useState(null); // null | 'indexing' | 'done' | 'error'
    const [message, setMessage] = useState('');
    const [indexStats, setIndexStats] = useState(null);

    function addFiles(newFiles) {
        const pdfs = [...newFiles].filter(f => f.name.toLowerCase().endsWith('.pdf'));
        setFiles(prev => {
            const existing = new Set(prev.map(f => f.name));
            return [...prev, ...pdfs.filter(f => !existing.has(f.name))];
        });
    }

    const onDrop = useCallback(e => {
        e.preventDefault();
        setDragging(false);
        addFiles(e.dataTransfer.files);
    }, []);

    async function handleIndex() {
        if (!files.length) return;
        setStatus('indexing');
        setMessage('');

        try {
            const result = await api.index(files);
            setIndexStats(result);
            setStatus('done');
            setMessage(`✅ Indexed ${result.num_chunks ?? '?'} chunks from ${result.num_files ?? files.length} file(s).`);
            onIndexed?.();
        } catch (err) {
            setStatus('error');
            setMessage(`❌ Indexing failed: ${err.message}`);
        }
    }

    return (
        <div className="upload-area">
            <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 4 }}>📄 Upload Documents</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: 13, marginBottom: 20 }}>
                Upload PDF files to index them for Q&amp;A
            </p>

            {/* Dropzone */}
            <div
                className={`dropzone ${dragging ? 'drag-over' : ''}`}
                onDragOver={e => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                onClick={() => document.getElementById('fileInput').click()}
            >
                <div className="dropzone-icon">☁️</div>
                <h3>Drag &amp; drop PDF files here</h3>
                <p>or click to browse · 200 MB per file</p>
                <input
                    id="fileInput"
                    type="file"
                    accept=".pdf"
                    multiple
                    hidden
                    onChange={e => addFiles(e.target.files)}
                />
            </div>

            {/* File list */}
            {files.length > 0 && (
                <div className="file-list" style={{ marginTop: 16 }}>
                    {files.map((f, i) => (
                        <div key={i} className="file-item">
                            <span style={{ fontSize: 18 }}>📄</span>
                            <span className="file-item-name">{f.name}</span>
                            <span>{formatBytes(f.size)}</span>
                            <button
                                className="file-remove"
                                onClick={() => setFiles(prev => prev.filter((_, j) => j !== i))}
                            >×</button>
                        </div>
                    ))}
                </div>
            )}

            {/* Progress */}
            {status === 'indexing' && (
                <>
                    <div className="progress-bar" style={{ marginTop: 16 }}>
                        <div className="progress-fill" style={{ width: '100%' }} />
                    </div>
                    <p style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 8 }}>
                        Chunking, embedding, and building index…
                    </p>
                </>
            )}

            {/* Status message */}
            {message && (
                <div className={`alert ${status === 'done' ? 'alert-success' : 'alert-error'}`}>
                    {message}
                </div>
            )}

            {/* Index button */}
            <button
                className="btn btn-primary btn-full"
                style={{ marginTop: 16 }}
                onClick={handleIndex}
                disabled={!files.length || status === 'indexing'}
            >
                {status === 'indexing' ? '⏳ Indexing…' : '🚀 Index Documents'}
            </button>

            {/* Stats after indexing */}
            {indexStats && (
                <div className="stats-grid">
                    <div className="stat-card">
                        <div className="stat-value">{indexStats.num_files ?? '?'}</div>
                        <div className="stat-label">Files indexed</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-value">{indexStats.num_chunks ?? '?'}</div>
                        <div className="stat-label">Total chunks</div>
                    </div>
                    {indexStats.embedding_model && (
                        <div className="stat-card" style={{ gridColumn: '1 / -1' }}>
                            <div className="stat-value" style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-secondary)' }}>
                                {indexStats.embedding_model}
                            </div>
                            <div className="stat-label">Embedding model</div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
