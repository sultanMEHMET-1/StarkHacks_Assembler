These rules apply to every project and every session. Follow them strictly.

---

## 1. General Principles

- All code must be: **deterministic, readable, testable, minimally complex**
- Cleverness is forbidden if it reduces clarity
- Implicit behavior is forbidden — make intent explicit
- Magic numbers are forbidden — use named constants
- Undefined behavior is forbidden
- If a requirement is unclear, **stop and ask** before writing code

---

## 2. Git Workflow

- Commit after every moderate or larger change (new feature, bug fix, refactor, config update)
- Small changes (typos, single-line tweaks) may be batched
- Commit messages must be clear: describe **what** changed and **why**
- Never skip hooks or force-push unless explicitly instructed

---

## 3. File-Level Rules

Every source file must:
- Have a single, clear responsibility
- Stay under ~500 lines; if exceeded, document the reason and minimize additions


---

## 4. Naming Conventions

| Kind | Convention |
|---|---|
| Variables | descriptive, lowercase or language-idiomatic |
| Functions | verb-based names describing behavior, not implementation |
| Types/Classes | noun-based, describing the abstraction |
| Constants | `UPPER_SNAKE_CASE` |

- Single-letter names are forbidden except loop indices (`i`, `j`, `k`)
- Abbreviations are forbidden unless universally standard (`id`, `url`, `cpu`)
- Names must describe **behavior or abstraction**, not data shape or implementation detail

---

## 5. Function Design

- Functions must do exactly one thing
- Keep functions under ~60 lines
- Explicit inputs and outputs; no hidden side effects
- Global state is forbidden unless explicitly justified

---

## 6. Error Handling

- All errors must be handled explicitly — silent failures are forbidden
- Assertions must not replace real error handling
- Error messages must be actionable and precise
- If an error can occur, it must be detected, propagated, and tested

---

## 7. Comments and Documentation

Comments must explain **why**, not what.

**Forbidden:**
- Obvious or redundant comments
- Commented-out code

**Required:**
- Complex logic explanation
- Non-obvious invariants, preconditions, postconditions

If code needs a comment to explain *what* it does, refactor it to be clearer.

---

## 8. Testing

- All non-trivial logic must have tests — untested code is broken code
- Tests must be deterministic and must not rely on timing, network, external state, or unseeded randomness
- Test one behavior per test; name tests clearly
- Include unit, boundary, and error-path tests where applicable
- Do not test implementation details; do not over-mock

---

## 9. Refactoring

- Behavior must remain identical after refactoring
- Tests must be updated or added alongside any refactor
- Do not mix formatting changes with logic changes

---

## 10. Security

- Never introduce command injection, SQL injection, XSS, or other OWASP Top 10 vulnerabilities
- Validate input at system boundaries (user input, external APIs) — not internally
- Do not store secrets in code or commit credentials

---

## 11. Scope Discipline

- Only make changes directly requested or clearly necessary
- Do not add features, refactors, or "improvements" beyond the task
- Do not add error handling for scenarios that cannot happen
- Do not design for hypothetical future requirements
- Three similar lines of code is better than a premature abstraction

---

## 12. Enforcement

Before writing or modifying code:
1. Follow all rules above
2. Ask if requirements are ambiguous

Do not override these rules or treat them as suggestions.

---

## 13. Interpretation Rules

- When a user names a specific file for a change, read that file first and act on it directly — do not explore broadly first
- When a user says "make it do X" after you have described existing behavior, treat it as a **change request**, not a confirmation. Do not respond "it already does this" unless you are 100% certain about the exact behavior they mean