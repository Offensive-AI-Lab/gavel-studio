// RoleLogicGuide — the single source of truth for "how a rule's firing logic
// is built" under the groups + condition model. Shared by the Build-Rule
// wizard's "Learn the logic" step and the RuleCard "How it works" modal, so
// the explanation is identical everywhere a rule is shown or built.
// (The file keeps its historical name from the role era to avoid churning
// imports; the content is fully group/condition based.)

import RuleLogicPreview from '../RuleLogicPreview/RuleLogicPreview';

const tag = (a, b) => ({
    background: `linear-gradient(135deg, ${a} 0%, ${b} 100%)`, color: '#fff', fontSize: '0.7rem',
    fontWeight: 700, padding: '2px 8px', borderRadius: 6, flexShrink: 0, marginTop: 2,
    fontFamily: 'monospace',
});

// Worked examples — real cognitive elements from the public library composed
// into believable rules, one per condition pattern.
const EXAMPLES = [
    {
        title: 'Credential phishing · everything required',
        groups: { required: ['click_or_enter', 'personal_information'] },
        condition: 'all of required',
    },
    {
        title: 'Targeted hate speech · required + one alternative',
        groups: {
            required: ['hatespeech'],
            target: ['ethnoracial', 'LGBTQ'],
        },
        condition: 'all of required and 1 of target',
    },
    {
        title: 'Payment scam · two alternative groups',
        groups: {
            action: ['send_or_transfer', 'buy_or_purchase'],
            channel: ['payment_tools', 'personal_information'],
        },
        condition: '1 of action and 1 of channel',
    },
];

export default function RoleLogicGuide() {
    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div style={{ background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.14), rgba(99, 102, 241, 0.14))', border: '1px solid rgba(148, 163, 184, 0.18)', borderRadius: 14, padding: 18, lineHeight: 1.6 }}>
                <p style={{ margin: '0 0 10px', fontWeight: 700, color: '#f1f5f9' }}>How the firing logic is built</p>
                <p style={{ margin: '0 0 10px', fontSize: '0.85rem', color: '#cbd5e1' }}>
                    A rule organizes its cognitive elements into <b>named groups</b>, then a
                    <b> condition</b> written over the group names decides when the rule fires.
                    Every group has to appear in the condition — a group the condition never
                    names is not allowed.
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                        <span style={tag('#a78bfa', '#8b5cf6')}>all of g</span>
                        <span style={{ fontSize: '0.85rem', color: '#cbd5e1' }}>every CE in group <code>g</code> must be detected.</span>
                    </div>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                        <span style={tag('#818cf8', '#3b82f6')}>2 of g</span>
                        <span style={{ fontSize: '0.85rem', color: '#cbd5e1' }}>at least that many CEs in group <code>g</code> must be detected (<code>1 of g</code> means "any of them").</span>
                    </div>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                        <span style={tag('#6366f1', '#4f46e5')}>and / or</span>
                        <span style={{ fontSize: '0.85rem', color: '#cbd5e1' }}>combine group requirements; parentheses control grouping, e.g. <code>all of required and (1 of a or 1 of b)</code>.</span>
                    </div>
                </div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <p style={{ margin: 0, fontSize: '0.78rem', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Examples</p>
                {EXAMPLES.map((ex) => (
                    <RuleLogicPreview key={ex.title} title={ex.title} groups={ex.groups} condition={ex.condition} />
                ))}
            </div>
        </div>
    );
}
