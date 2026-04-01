from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class EligibilityResult:
    eligible: bool
    reason: str
    emi: int
    ltv_percent: float
    foir_percent: float
    max_loan_by_ltv: int
    max_emi_by_foir: int
    max_loan_by_foir: int
    recommended_max_loan: int

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_emi(principal: float, annual_rate: float = 9.2, years: int = 20) -> int:
    """Standard EMI formula with safe defaults for a home-loan estimate."""
    if principal <= 0:
        return 0
    if annual_rate <= 0:
        return int(round(principal / (years * 12)))

    monthly_rate = (annual_rate / 12) / 100
    months = years * 12
    emi = (principal * monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
    return int(round(emi))


def _max_loan_from_emi(target_emi: float, annual_rate: float, years: int) -> int:
    if target_emi <= 0:
        return 0
    monthly_rate = (annual_rate / 12) / 100
    months = years * 12
    if monthly_rate == 0:
        return int(target_emi * months)
    factor = ((1 + monthly_rate) ** months - 1) / (monthly_rate * (1 + monthly_rate) ** months)
    return int(round(target_emi * factor))


def check_eligibility(
    monthly_income: float,
    loan_amount: float,
    property_value: float,
    employment_status: str = "unknown",
    annual_rate: float = 9.2,
    years: int = 20,
) -> EligibilityResult:
    """Deterministic rule engine using simple LTV and FOIR policy checks.

    Rules used in this prototype:
    - LTV cap: 80%
    - FOIR cap: 50% for salaried, 45% for self-employed, 45% default
    """
    if monthly_income <= 0 or loan_amount <= 0 or property_value <= 0:
        return EligibilityResult(
            eligible=False,
            reason="Invalid inputs. monthly_income, loan_amount and property_value must be > 0.",
            emi=0,
            ltv_percent=0.0,
            foir_percent=0.0,
            max_loan_by_ltv=0,
            max_emi_by_foir=0,
            max_loan_by_foir=0,
            recommended_max_loan=0,
        )

    status = (employment_status or "").strip().lower()
    foir_limit = 50.0 if "salar" in status else 45.0 if "self" in status else 45.0
    ltv_limit = 80.0

    ltv_percent = (loan_amount / property_value) * 100
    emi = calculate_emi(loan_amount, annual_rate=annual_rate, years=years)
    foir_percent = (emi / monthly_income) * 100

    max_loan_by_ltv = int(property_value * (ltv_limit / 100))
    max_emi_by_foir = int(monthly_income * (foir_limit / 100))
    max_loan_by_foir = _max_loan_from_emi(max_emi_by_foir, annual_rate=annual_rate, years=years)
    recommended_max_loan = min(max_loan_by_ltv, max_loan_by_foir)

    if ltv_percent > ltv_limit:
        return EligibilityResult(
            eligible=False,
            reason=f"Rejected due to LTV. Requested {ltv_percent:.1f}% vs limit {ltv_limit:.0f}%.",
            emi=emi,
            ltv_percent=round(ltv_percent, 2),
            foir_percent=round(foir_percent, 2),
            max_loan_by_ltv=max_loan_by_ltv,
            max_emi_by_foir=max_emi_by_foir,
            max_loan_by_foir=max_loan_by_foir,
            recommended_max_loan=recommended_max_loan,
        )

    if foir_percent > foir_limit:
        return EligibilityResult(
            eligible=False,
            reason=f"Rejected due to FOIR. Estimated FOIR {foir_percent:.1f}% vs limit {foir_limit:.0f}%.",
            emi=emi,
            ltv_percent=round(ltv_percent, 2),
            foir_percent=round(foir_percent, 2),
            max_loan_by_ltv=max_loan_by_ltv,
            max_emi_by_foir=max_emi_by_foir,
            max_loan_by_foir=max_loan_by_foir,
            recommended_max_loan=recommended_max_loan,
        )

    return EligibilityResult(
        eligible=True,
        reason="Eligible under current prototype policy checks.",
        emi=emi,
        ltv_percent=round(ltv_percent, 2),
        foir_percent=round(foir_percent, 2),
        max_loan_by_ltv=max_loan_by_ltv,
        max_emi_by_foir=max_emi_by_foir,
        max_loan_by_foir=max_loan_by_foir,
        recommended_max_loan=recommended_max_loan,
    )