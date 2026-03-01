import { useState } from 'react';

const PANELS = [
    { id: 'chat', label: 'Chat', icon: '💬', desc: 'Ask questions about your docs' },
    { id: 'upload', label: 'Upload & Index', icon: '📄', desc: 'Upload & index PDF files' },
    { id: 'evaluation', label: 'Evaluation', icon: '📊', desc: 'View metrics & analytics' },
];

/**
 * StackLayout
 * Props:
 *   active        – current panel id
 *   onSelect(id)  – called when user picks a different card
 *   children      – render-prop function: (id) => JSX
 */
export default function StackLayout({ active, onSelect, children }) {
    const [animating, setAnimating] = useState(false);

    // Build ordered list: active first, then the rest in original order
    const ordered = [
        PANELS.find(p => p.id === active),
        ...PANELS.filter(p => p.id !== active),
    ];

    function handleSelect(id) {
        if (id === active || animating) return;
        setAnimating(true);
        onSelect(id);
        setTimeout(() => setAnimating(false), 600);
    }

    return (
        <div className="stack-root">
            {/* Stack scene */}
            <div className="stack-scene">
                {ordered.map((panel, stackIdx) => {
                    const isActive = stackIdx === 0;
                    return (
                        <div
                            key={panel.id}
                            className={`stack-card ${isActive ? 'stack-card--active' : 'stack-card--back'} stack-pos-${stackIdx}`}
                            style={{ '--stack-idx': stackIdx }}
                            onClick={() => !isActive && handleSelect(panel.id)}
                        >
                            {/* Card header strip (always visible even when stacked) */}
                            <div className={`stack-card-header ${isActive ? 'active' : ''}`}>
                                <span className="stack-card-icon">{panel.icon}</span>
                                <span className="stack-card-label">{panel.label}</span>
                                {!isActive && (
                                    <span className="stack-card-desc">{panel.desc}</span>
                                )}
                                {isActive && (
                                    <div className="stack-card-pills">
                                        {PANELS.filter(p => p.id !== active).map(p => (
                                            <button
                                                key={p.id}
                                                className="stack-pill-btn"
                                                onClick={e => { e.stopPropagation(); handleSelect(p.id); }}
                                            >
                                                {p.icon} {p.label}
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>

                            {/* Card body – only rendered for active */}
                            {isActive && (
                                <div className="stack-card-body">
                                    {children(panel.id)}
                                </div>
                            )}

                            {/* Decorative scanline for futuristic feel */}
                            {!isActive && (
                                <div className="stack-card-scanline" />
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
