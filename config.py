APP_BUILD = "2026-02-02-03"

DEFAULT_WINDOW_DAYS = 30
DEFAULT_MAXRECORDS = 250

DEFAULT_QUERY = (
    '(United States OR USA OR US OR Pentagon OR CENTCOM) '
    '(Iran OR Iranian OR Tehran OR IRGC OR "Strait of Hormuz" OR "Persian Gulf")'
)

# ----- Keywords (GDELT-only intent model) -----
ESCALATION_KW = [
    "airstrike", "strike", "missile", "rocket", "drone", "ballistic",
    "retaliat", "revenge", "attack", "assault", "intercept", "downed",
    "irgc", "revolutionary guard", "sanction", "embargo",
    "detain", "arrest", "seiz", "confiscat",
    "naval", "warship", "destroyer", "carrier", "submarine",
    "proxy", "militia", "hezbollah", "houthis",
    "explosion", "blast", "killed", "casualt",
    "nuclear", "uranium", "enrichment",
    "strait of hormuz", "persian gulf",
]

DEESCALATION_KW = [
    "talk", "talks", "negotiat", "mediat", "diplomac", "peace", "ceasefire",
    "agreement", "deal", "backchannel", "dialogue", "de-escalat", "deescalat",
    "confidence-building", "summit",
]

MILITARY_KW = [
    "airstrike", "strike", "missile", "rocket", "drone", "ballistic",
    "attack", "assault", "bomb", "explosion", "blast", "killed", "casualt",
    "intercept", "downed", "naval", "warship", "destroyer", "carrier",
    "military", "exercise", "mobiliz", "troop", "deployment",
]

ECONOMIC_KW = [
    "sanction", "embargo", "designat", "terrorist designation",
    "oil", "brent", "wti", "gas", "lng",
    "shipping", "tanker", "container", "reroute", "insurance",
    "strait of hormuz", "hormuz", "red sea",
    "inflation", "currency", "fx", "rial",
]

# Theme needles (only used if your feed provides themes_list)
CONFLICT_THEME_NEEDLES = [
    "ARMEDCONFLICT", "TERRORISM", "MILITARY", "VIOLENCE", "SECURITYSERVICES",
    "WEAPONS", "AIRSTRIKES", "MISSILES", "DRONE", "NAVY", "SANCTIONS",
]

DIPLO_THEME_NEEDLES = [
    "NEGOTIATIONS", "MEDIATION", "DIPLOMACY", "PEACE", "CEASEFIRE", "TREATY",
]
