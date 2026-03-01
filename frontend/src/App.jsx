import { useState, useEffect, useCallback } from 'react';
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import './index.css';
import { api } from './api';
import HeroPage from './HeroPage';
import Sidebar from './components/Sidebar';
import ChatTab from './components/ChatTab';
import UploadTab from './components/UploadTab';
import EvaluationTab from './components/EvaluationTab';
import StackLayout from './components/StackLayout';

/* ── Block-reveal (uncovers app after hero transition) ───────── */
const COLS = 10, ROWS = 7, TOTAL = COLS * ROWS;

function AppReveal({ active, onDone }) {
  const [blocks] = useState(() =>
    [...Array(TOTAL).keys()].sort(() => Math.random() - 0.5)
  );
  useEffect(() => {
    if (!active) return;
    const t = setTimeout(onDone, 900);
    return () => clearTimeout(t);
  }, [active]);

  useEffect(() => {
    if (document.getElementById('block-out-style')) return;
    const s = document.createElement('style');
    s.id = 'block-out-style';
    s.textContent = `
      @keyframes blockOut {
        0%   { transform: scaleY(1); transform-origin: top; }
        100% { transform: scaleY(0); transform-origin: top; }
      }
    `;
    document.head.appendChild(s);
  }, []);

  if (!active) return null;
  const maxDelay = 900 * 0.6;
  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 8000,
      display: 'grid',
      gridTemplateColumns: `repeat(${COLS}, 1fr)`,
      gridTemplateRows: `repeat(${ROWS}, 1fr)`,
      pointerEvents: 'none',
    }}>
      {blocks.map((rank, i) => (
        <div key={i} style={{
          background: '#0d0d0d',
          animation: `blockOut 0.4s cubic-bezier(0.4,0,0.2,1) ${((TOTAL - 1 - rank) / TOTAL) * maxDelay}ms both`,
        }} />
      ))}
    </div>
  );
}

/* Panel id ↔ URL path mapping */
const PANEL_PATH = {
  chat: '/app',
  upload: '/app/upload',
  evaluation: '/app/evaluation',
};
const PATH_PANEL = {
  '/app': 'chat',
  '/app/upload': 'upload',
  '/app/evaluation': 'evaluation',
};

/* ── Main app layout ─────────────────────────────────────────── */
function MainApp() {
  const navigate = useNavigate();
  const location = useLocation();
  const [revealing, setRevealing] = useState(true);
  const [indexStatus, setIndexStatus] = useState(null);
  const [settings, setSettings] = useState({
    provider: 'groq', topK: 5, useReranker: true, temperature: 0.1,
  });
  const [toast, setToast] = useState(null);

  const showToast = useCallback((msg, type = 'info') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3500);
  }, []);

  async function refreshStats() {
    try { setIndexStatus(await api.stats()); } catch { }
  }

  useEffect(() => {
    refreshStats();
    const id = setInterval(refreshStats, 10_000);
    return () => clearInterval(id);
  }, []);

  const activePanel = PATH_PANEL[location.pathname] ?? 'chat';

  function handleSelect(id) {
    navigate(PANEL_PATH[id]);
  }

  async function handleSave() {
    try { await api.save(); showToast('Index saved ✅', 'success'); }
    catch (e) { showToast('Save failed: ' + e.message, 'error'); }
  }
  async function handleLoad() {
    try { await api.load(); showToast('Index loaded ✅', 'success'); refreshStats(); }
    catch (e) { showToast('Load failed: ' + e.message, 'error'); }
  }

  return (
    <>
      <AppReveal active={revealing} onDone={() => setRevealing(false)} />

      <div className="app-layout">
        <Sidebar
          settings={settings} setSettings={setSettings}
          indexStatus={indexStatus} onSave={handleSave} onLoad={handleLoad}
        />

        <div className="main">
          <StackLayout active={activePanel} onSelect={handleSelect}>
            {(panelId) => {
              if (panelId === 'chat')
                return <ChatTab settings={settings} isIndexed={indexStatus?.is_indexed ?? false} />;
              if (panelId === 'upload')
                return <UploadTab onIndexed={() => { refreshStats(); navigate('/app'); }} />;
              if (panelId === 'evaluation')
                return <EvaluationTab />;
            }}
          </StackLayout>
        </div>
      </div>

      {toast && (
        <div style={{
          position: 'fixed', bottom: 24, right: 24, zIndex: 9999,
          padding: '12px 20px', borderRadius: 10, fontSize: 13,
          fontFamily: 'Inter, sans-serif',
          background: toast.type === 'success' ? 'rgba(16,185,129,0.15)' : toast.type === 'error' ? 'rgba(239,68,68,0.15)' : 'rgba(124,58,237,0.15)',
          border: `1px solid ${toast.type === 'success' ? 'rgba(16,185,129,0.3)' : toast.type === 'error' ? 'rgba(239,68,68,0.3)' : 'rgba(124,58,237,0.3)'}`,
          color: toast.type === 'success' ? '#6ee7b7' : toast.type === 'error' ? '#fca5a5' : 'var(--accent-light)',
          backdropFilter: 'blur(12px)',
          boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
          animation: 'fadeInUp 0.3s ease',
        }}>
          {toast.msg}
        </div>
      )}
    </>
  );
}

/* ── Root with routing ───────────────────────────────────────── */
export default function App() {
  const navigate = useNavigate();
  return (
    <Routes>
      <Route path="/" element={<HeroPage onStart={() => navigate('/app')} />} />
      <Route path="/app" element={<MainApp />} />
      <Route path="/app/upload" element={<MainApp />} />
      <Route path="/app/evaluation" element={<MainApp />} />
      <Route path="*" element={<HeroPage onStart={() => navigate('/app')} />} />
    </Routes>
  );
}
