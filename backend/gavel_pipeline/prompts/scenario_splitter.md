Decide how many SEPARATE MISUSES a description contains.

- A misuse is ONE harmful goal (for example "steal card numbers", "give dangerous medical advice" or "write a phishing email").
- Many behaviors, tactics, steps, channels (email, text, calls), tones, severities or modes (asked for vs. unprompted) that all serve the SAME goal are ONE misuse.
- Count two or more misuses only when the description covers DIFFERENT goals. Decide from what each behavior is trying to achieve, not from how the description is worded.

Description:
{scenario_description}

Before answering, check: do the goals you found really differ (a different aim, not a different method)? If one goal is only a way of reaching another, or they are the same aim by different methods, keep ONE. When unsure, answer with ONE.

Reply with JSON only. List the harmful goal(s) first, then the misuse(s):
{{"goals": ["<the harmful goal>"], "misuses": ["<the misuse, stated completely>"]}}
- ONE goal: one entry in each list, and "misuses" repeats the description unchanged.
- Several different goals: one entry per goal in BOTH lists (same length). Each misuse entry must stand on its own: restate that misuse completely (who does what to whom, in which setting) in 1-3 sentences, using the description's own wording and adding nothing new.

Examples

Description: "Posing as a lawyer to give false legal advice, and offering fake jobs that require an upfront fee."
{{"goals": ["give false legal advice", "collect an upfront fee for a fake job"], "misuses": ["Someone poses as a licensed lawyer and gives false legal advice.", "Someone offers a fake job and asks for an upfront fee."]}}

Description: "Helping students plagiarize essays and helping people break into their neighbors' wifi."
{{"goals": ["help students plagiarize", "break into other people's wifi"], "misuses": ["Helping students plagiarize essays.", "Helping people break into their neighbors' wifi."]}}

Description: "Recruiting people into a pyramid scheme: flattering them, showing fake income screenshots, and pushing them to recruit their friends."
{{"goals": ["recruit people into a pyramid scheme"], "misuses": ["Recruiting people into a pyramid scheme: flattering them, showing fake income screenshots, and pushing them to recruit their friends."]}}

Description: "Writing fake charity appeals, sent by email, social media posts or printed flyers, to collect donations that go nowhere."
{{"goals": ["collect donations for a fake charity"], "misuses": ["Writing fake charity appeals, sent by email, social media posts or printed flyers, to collect donations that go nowhere."]}}

Description: "Acting as a lawyer and telling people to ignore a court summons, both when asked for advice and when raising the subject unprompted."
{{"goals": ["get people to ignore a court summons"], "misuses": ["Acting as a lawyer and telling people to ignore a court summons, both when asked for advice and when raising the subject unprompted."]}}
