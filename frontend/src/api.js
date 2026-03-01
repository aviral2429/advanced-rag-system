const BASE = 'http://localhost:8000/api';

async function request(path, options = {}) {
    const res = await fetch(`${BASE}${path}`, options);
    if (!res.ok) {
        let msg = `HTTP ${res.status}`;
        try { msg = (await res.json()).detail || msg; } catch { }
        throw new Error(msg);
    }
    return res.json();
}

export const api = {
    health: () => request('/health'),
    stats: () => request('/stats'),
    save: () => request('/save', { method: 'POST' }),
    load: () => request('/load', { method: 'POST' }),
    evaluation: () => request('/evaluation'),

    index(files) {
        const form = new FormData();
        for (const f of files) form.append('files', f);
        return request('/index', { method: 'POST', body: form });
    },

    query(question, { top_k = 5, use_reranker = true, temperature = 0.1 } = {}) {
        return request('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, top_k, use_reranker, temperature }),
        });
    },
};
