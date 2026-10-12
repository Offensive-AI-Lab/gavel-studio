Two parts of a description of misuse are below. Are they two steps of ONE scheme, or two INDEPENDENT problems?

Description they came from: {scenario_description}
A: {part_a}
B: {part_b}

- SAME_SCHEME: A and B work together toward the same single aim; one is a step, tactic or channel of the other (for example threatening someone and then demanding payment).
- INDEPENDENT: A and B have different aims. Either could happen without the other, in different conversations, against different victims. This holds EVEN IF both happen in the same company, product or domain.

First write one or two sentences comparing what A and B each try to achieve, then reply with JSON only: {{"a_aim": "<aim of A>", "b_aim": "<aim of B>", "relation": "SAME_SCHEME"}} or {{"a_aim": "<aim of A>", "b_aim": "<aim of B>", "relation": "INDEPENDENT"}}

Examples

A: "Posing as a lawyer and giving false legal advice."
B: "Offering a fake job and asking for an upfront fee."
{{"a_aim": "mislead people about the law", "b_aim": "collect a fee for a fake job", "relation": "INDEPENDENT"}}

A: "A hotel-booking assistant refuses guests because of their nationality."
B: "The same hotel-booking assistant sells copies of guests' passports."
{{"a_aim": "exclude guests by nationality", "b_aim": "profit from guests' private documents", "relation": "INDEPENDENT"}}

A: "Sending a fake bank alert that creates panic about a stolen account."
B: "Then asking the victim to read out the one-time code."
{{"a_aim": "steal access to the account", "b_aim": "steal access to the account", "relation": "SAME_SCHEME"}}

A: "Writing fake charity appeals as emails."
B: "Writing fake charity appeals as social media posts."
{{"a_aim": "collect money for a fake charity", "b_aim": "collect money for a fake charity", "relation": "SAME_SCHEME"}}
