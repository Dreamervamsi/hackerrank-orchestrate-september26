# Execution Examples & Edge-Case Verifications

This document showcases concrete execution examples and outlines how the hybrid architecture handles edge cases cleanly.

---

## Example 1: `request_26` — Clear-cut Immediate Affordability
- **User ID:** `user_26`
- **Liquid Balance:** `100,845,250.00 IDR`
- **Min Balance to Keep:** `24,768,300.00 IDR`
- **Requested Amount:** `15,656,000.00 IDR`
- **Workflow:** Bypasses LLM call entirely via the **Zero-LLM Fast Path**.
- **Simulator Output:** 
  - Starting available balance minus requested amount: `85,189,250.00 IDR`.
  - Running daily balance over 90 days stays strictly above `24,768,300.00 IDR`.
- **Result:**
  - **Status:** `affordable_now`
  - **Method:** `full_payment`
  - **Plan:** `2025-08-03:15656000` (request_date:amount)
  - **Validation:** **PASSED**

---

## Example 2: `request_28` — Waitable Future Affordability
- **User ID:** `user_28`
- **Liquid Balance:** `1,789.40 EUR`
- **Min Balance to Keep:** `1,100.00 EUR`
- **Requested Amount:** `1,302.40 EUR`
- **Workflow:** Deterministic pre-fetching and simulation.
- **Simulator Output:**
  - Subtracting `1,302.40 EUR` from `1,789.40 EUR` leaves `487.00 EUR`, which violates the `1,100.00 EUR` minimum balance.
  - The simulator scans ahead day-by-day and finds that on **2024-08-15**, the user receives a confirmed salary payout, bringing their balance comfortably high enough to support the full lump-sum payment safely.
- **Result:**
  - **Status:** `affordable_later`
  - **Method:** `wait`
  - **Plan:** `2024-08-15:1302.40` (earliest_date:amount)
  - **Validation:** **PASSED**

---

## Example 3: `request_32` — API Token Exhaustion Resiliency
- **User ID:** `user_32`
- **Condition:** Executed while Groq Cloud's daily TPD (Tokens Per Day) allowance is fully depleted.
- **Workflow:** The LLM client call throws `rate_limit_exceeded (429)`.
- **Simulator Output:**
  - The try-except wrapper catches the rate-limit exception.
  - The **Resilient Safe Fallback Engine** takes over, queries our indexed baseline database, runs the daily simulation, finds the earliest safe payment date, and completes the row cleanly in Python.
- **Result:**
  - **Status:** `affordable_later`
  - **Method:** `wait`
  - **Plan:** `2025-02-15:40018`
  - **Validation:** **PASSED** (100% compliant)
