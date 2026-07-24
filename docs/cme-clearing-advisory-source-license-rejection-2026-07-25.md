# CME Clearing Advisory Source License Rejection

## Decision

Reject CME clearing and performance-bond advisories as an automated LLM/ML
alpha source unless the repository later obtains explicit written permission
from CME Group.

The historical notices are economically relevant and date-coded, but current
CME terms expressly prohibit machine-learning or artificial-intelligence use
of website content and prohibit automated bulk collection, systematic
retrieval, text mining, and dataset compilation without permission.

## Official evidence

- CME Website Terms of Use:
  <https://www.cmegroup.com/files/terms-of-use.pdf>
- CME Data Terms of Use:
  <https://www.cmegroup.com/trading/market-data-explanation-disclaimer.html>
- Notice archive description:
  <https://www.cmegroup.com/notices.html>

The restriction defeats both required operations for this project:

1. building a reproducible historical advisory ledger; and
2. using advisory text or tables with an LLM/ML policy.

Public browser visibility does not grant those rights.

## Prohibited repair

Do not scrape advisory indexes or PDFs, enumerate notice IDs, train or prompt a
model with notice content, derive performance-bond features, or retain a local
advisory corpus. Search-engine snippets and third-party mirrors do not cure the
license boundary.

## Outcome boundary

No CME advisory corpus, BTC outcome, model training row, prompt dataset, or
performance result was created. This is a license rejection before source or
alpha evaluation.
