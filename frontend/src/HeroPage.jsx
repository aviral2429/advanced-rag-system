import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Orb from './Orb';

/* ── Block-cover transition on hero exit ────────────────────── */
const COLS = 10, ROWS = 7, TOTAL = COLS * ROWS;
const DURATION = 850;

function BlockCover({ active, onDone }) {
    const [visible, setVisible] = useState(false);
    const [order] = useState(() =>
        [...Array(TOTAL).keys()].sort(() => Math.random() - 0.5)
    );

    useEffect(() => {
        if (!active) { setVisible(false); return; }
        setVisible(true);
        const t = setTimeout(onDone, DURATION + 80);
        return () => clearTimeout(t);
    }, [active]);

    if (!visible) return null;
    const maxDelay = DURATION * 0.65;

    return (
        <div style={{
            position: 'fixed', inset: 0, zIndex: 9999,
            display: 'grid',
            gridTemplateColumns: `repeat(${COLS}, 1fr)`,
            gridTemplateRows: `repeat(${ROWS}, 1fr)`,
            pointerEvents: 'none',
        }}>
            {order.map((rank, i) => (
                <div key={i} style={{
                    background: '#0d0d0d',
                    animation: `blockIn 0.32s cubic-bezier(0.4,0,0.2,1) ${(rank / TOTAL) * maxDelay}ms both`,
                }} />
            ))}
        </div>
    );
}

/* ── Cycling taglines ─────────────────────────────────────────── */
const TAGLINES = [
    "Drop your PDFs. Ask anything. We'll handle the chaos.",
    "Finally — AI that actually reads your documents.",
    "Your PDFs have secrets. Let's crack 'em open. 🔍",
    "Smarter than ctrl+F. Way smarter.",
    "Stop scrolling. Start asking. ✨",
];

export default function HeroPage({ onStart }) {
    const navigate = useNavigate();
    const [covering, setCovering] = useState(false);
    const [taglineIdx, setTaglineIdx] = useState(0);
    const [taglineFade, setTaglineFade] = useState(true);

    // Cycle taglines every 3s
    useEffect(() => {
        const id = setInterval(() => {
            setTaglineFade(false);
            setTimeout(() => {
                setTaglineIdx(i => (i + 1) % TAGLINES.length);
                setTaglineFade(true);
            }, 300);
        }, 3000);
        return () => clearInterval(id);
    }, []);

    function handleStart() {
        setCovering(true);
    }

    function handleDone() {
        if (onStart) onStart();
        navigate('/app');
    }

    return (
        <>
            <BlockCover active={covering} onDone={handleDone} />

            <div className="hero-page hero-centered">
                {/* Fullscreen animated Orb */}
                <div className="hero-orb-fullscreen">
                    <Orb
                        hue={220}
                        hoverIntensity={5}
                        rotateOnHover
                        forceHoverState={false}
                        backgroundColor="#0d0d0d"
                    />
                </div>

                {/* Centered overlay content */}
                <div className="hero-overlay-content">
                    <h1 className="hero-title">
                        Advanced&nbsp;<span className="hero-title-accent">RAG</span>&nbsp;System
                    </h1>

                    {/* Animated cycling tagline */}
                    <p
                        className="hero-tagline"
                        style={{ opacity: taglineFade ? 1 : 0, transition: 'opacity 0.3s ease' }}
                    >
                        {TAGLINES[taglineIdx]}
                    </p>

                    {/* CTA button */}
                    <button
                        className="hero-cta-btn"
                        onClick={handleStart}
                    >
                        <span className="hero-cta-shimmer" />
                        <span className="hero-cta-text">
                            Get Started
                            <span className="hero-cta-icon">→</span>
                        </span>
                    </button>

                    {/* Subtle hint */}
                    <p className="hero-hint">No sign-up · Open source · Runs locally</p>
                </div>
            </div>
        </>
    );
}
