from dataclasses import dataclass


@dataclass(frozen=True)
class OfferRule:
    code: str
    label: str


OFFERS = [
    OfferRule("motor_addon", "Motor Add-on Cover"),
    OfferRule("family_health", "Family Floater Health Plan"),
    OfferRule("home_insurance", "Home Insurance"),
    OfferRule("travel_insurance", "Travel Insurance"),
    OfferRule("critical_illness", "Critical Illness Rider"),
    OfferRule("wellness_app", "Wellness App Activation"),
    OfferRule("renewal_outreach", "Renewal Retention Action"),
]
