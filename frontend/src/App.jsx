import { useState, useEffect, useCallback } from 'react';
import './index.css';
import { api } from './api';
import Sidebar from './components/Sidebar';
import ChatTab from './components/ChatTab';
import UploadTab from './components/UploadTab';
import EvaluationTab from './components/EvaluationTab';

const TABS = [
  { id: 'chat', label: '💬 Chat' },
  { id: 'upload', label: '📄 Upload & Index' },
  { id: 'evaluation', label: '📊 Evaluation' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [indexStatus, setIndexStatus] = useState(null);
  const [settings, setSettings] = useState({
    provider: 'groq',
    topK: 5,
    useReranker: true,
    temperature: 0.1,
  });
  const [toast, setToast] = useState(null);

  const showToast = useCallback((msg, type = 'info') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3500);
  }, []);

  async function refreshStats() {
    try {
      const s = await api.stats();
      setIndexStatus(s);
    } catch { /* backend might not be up yet */ }
  }

  useEffect(() => {
    refreshStats();
    const id = setInterval(refreshStats, 10_000);
    return () => clearInterval(id);
  }, []);

  async function handleSave() {
    try { await api.save(); showToast('Index saved to disk ✅', 'success'); }
    catch (e) { showToast('Save failed: ' + e.message, 'error'); }
  }

  async function handleLoad() {
    try {
      const res = await api.load();
      showToast('Index loaded ✅', 'success');
      refreshStats();
    } catch (e) { showToast('Load failed: ' + e.message, 'error'); }
  }

  function handleIndexed() {
    refreshStats();
    setActiveTab('chat');
  }

  return (
    <div className="app-layout">
      <Sidebar
        settings={settings}
        setSettings={setSettings}
        indexStatus={indexStatus}
        onSave={handleSave}
        onLoad={handleLoad}
      />

      <div className="main">
        {/* Tabs bar */}
        <div className="tabs-bar">
          {TABS.map(t => (
            <button
              key={t.id}
              className={`tab-btn ${activeTab === t.id ? 'active' : ''}`}
              onClick={() => setActiveTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="tab-content">
          {activeTab === 'chat' && (
            <ChatTab
              settings={settings}
              isIndexed={indexStatus?.is_indexed ?? false}
            />
          )}
          {activeTab === 'upload' && (
            <UploadTab onIndexed={handleIndexed} />
          )}
          {activeTab === 'evaluation' && (
            <EvaluationTab />
          )}
        </div>
      </div>

      {/* Toast notification */}
      {toast && (
        <div
          style={{
            position: 'fixed', bottom: 24, right: 24, zIndex: 9999,
            padding: '12px 20px', borderRadius: 10, fontSize: 13,
            fontFamily: 'Inter, sans-serif',
            background: toast.type === 'success' ? 'rgba(16,185,129,0.15)' : toast.type === 'error' ? 'rgba(239,68,68,0.15)' : 'rgba(59,130,246,0.15)',
            border: `1px solid ${toast.type === 'success' ? 'rgba(16,185,129,0.3)' : toast.type === 'error' ? 'rgba(239,68,68,0.3)' : 'rgba(59,130,246,0.3)'}`,
            color: toast.type === 'success' ? '#6ee7b7' : toast.type === 'error' ? '#fca5a5' : 'var(--accent-light)',
            backdropFilter: 'blur(12px)',
            boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
            animation: 'fadeInUp 0.3s ease',
          }}
        >
          {toast.msg}
        </div>
      )}
    </div>
  );
}
