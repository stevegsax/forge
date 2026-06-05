---
requirement_id: inspira
title: Inspira inspirational quote page
status: approved
owner: example-team
reviewers:
  - example-reviewer
code_repo: inspira
test_repo: inspira-tests
paired_feature: docs/requirements/examples/inspira.feature
last_updated: 2026-06-05
language_target: python
architecture_pattern: functional-core-imperative-shell
---

# Inspira Inspirational Quote Page

## 1. Scope And Outcome

### Goal

Serve a web page at `GET /` that displays one inspirational quote chosen at random from a local
text file.

### In Scope

- read a configured UTF-8 quote file from the local filesystem
- parse the file into a validated quote catalog
- choose one quote per request using an injected random index source
- render an HTML page showing the selected quote
- return a user-friendly configuration error page when quotes cannot be loaded

### Out Of Scope

- author attribution
- quote categories, search, or filtering
- quote creation, editing, or deletion through the web UI
- client-side persistence of the previous quote
- any guarantee that two consecutive refreshes show different quotes
- database storage or internet-fetched quotes

### External Dependencies

- local filesystem read access to the configured quote file
- an HTTP server capable of handling `GET /`
- an injected random index source

## 2. Domain Algebra

### Entities And Value Objects

| Name | Kind | Definition | Illegal States Made Unrepresentable |
|------|------|------------|-------------------------------------|
| `QuoteText` | `value-object` | A trimmed, non-empty Unicode string representing one quote line. | Blank or whitespace-only quotes cannot exist as `QuoteText`. |
| `QuoteCatalog` | `value-object` | A non-empty ordered sequence of distinct `QuoteText` values. | An empty catalog or a catalog with duplicate quote texts cannot exist. |
| `QuoteIndex` | `value-object` | An integer `i` such that `0 <= i < len(catalog)`. | Negative indexes or indexes outside the catalog length cannot exist. |
| `QuotePage` | `value-object` | HTML document string containing exactly one rendered quote. | A page with zero or multiple displayed quotes cannot be produced by the pure renderer. |
| `CatalogError` | `enum` | `file_missing`, `file_unreadable`, `catalog_empty`, `duplicate_quote`, `random_index_out_of_range`. | Unclassified catalog failures are not represented in the core. |

### Derived Fields

| Field | Derived From | Why It Is Not Stored Independently |
|------|---------------|------------------------------------|
| `catalog_size` | `len(QuoteCatalog)` | It is fully derivable from the catalog and would risk inconsistency if stored separately. |

### State Machine

| From | Event | To | Notes |
|------|-------|----|-------|
| `request_started` | `quote file read succeeds` | `catalog_loaded` | The shell passes a raw text snapshot into the core. |
| `request_started` | `quote file read fails` | `configuration_error` | The shell maps file access failures to a typed `CatalogError`. |
| `catalog_loaded` | `catalog validates` | `quote_selected` | Selection requires a valid `QuoteCatalog` and `QuoteIndex`. |
| `catalog_loaded` | `catalog validation fails` | `configuration_error` | Empty or duplicate catalogs are rejected before selection. |
| `quote_selected` | `page rendered` | `response_ready` | Rendering is deterministic for a given `QuoteText`. |

## 3. Behavioral Examples

- Paired feature file: `docs/requirements/examples/inspira.feature`
- Scenario groups covered:
  - successful selection and rendering
  - repeated refresh behavior
  - file parsing rules
  - HTML escaping
  - configuration error handling
  - file update visibility on the next request
- Intentional gaps:
  - CSS and visual styling details are omitted because they do not affect the contract
  - deployment topology is omitted because the example focuses on requirements structure

## 4. Contract Map

| rule_id | intent | formal_statement | kind | strongest_layer | static_encoding_candidate | runtime_check | witnesses | counterexamples | failure_mode |
|---------|--------|------------------|------|-----------------|---------------------------|---------------|-----------|-----------------|--------------|
| `INSPIRA-RULE-001` | A quote is one meaningful line of text. | For every raw line `L`, `QuoteText` may be constructed iff `trim(L) != ""`; stored value is `trim(L)`. | `construction` | `construction` | `QuoteText.from_line(str) -> QuoteText | None` | `none` | `" Stay hungry, stay foolish. "` becomes `QuoteText("Stay hungry, stay foolish.")` | `""`, `"   "` | Blank lines do not produce a quote. |
| `INSPIRA-RULE-002` | The catalog must contain at least one usable quote. | `QuoteCatalog(quotes)` is valid iff `len(quotes) >= 1`. | `construction` | `construction` | `QuoteCatalog` wrapping a non-empty sequence | `none` | `["Stay hungry, stay foolish."]` | `[]` | Return `catalog_empty` and render HTTP 503 error page. |
| `INSPIRA-RULE-003` | Quote texts are unique after trimming. | `QuoteCatalog(quotes)` is valid iff `len(set(quotes)) = len(quotes)`. | `construction` | `construction` | `QuoteCatalog` constructor rejects duplicates | `none` | `["Stay hungry, stay foolish.", "The only way out is through."]` | `["Stay hungry, stay foolish.", " Stay hungry, stay foolish. "]` | Return `duplicate_quote` and render HTTP 503 error page. |
| `INSPIRA-RULE-004` | Quote selection chooses exactly one quote from the catalog. | Given valid `QuoteCatalog C` and valid `QuoteIndex i`, `select(C, i) = C[i]`. | `operation` | `operation` | `select_quote(catalog: QuoteCatalog, index: QuoteIndex) -> QuoteText` | `random_index_out_of_range` if shell violates contract | catalog `["A", "B", "C"]`, index `1` returns `"B"` | index `3` for catalog size `3` | Reject request as configuration failure and render HTTP 503 error page. |
| `INSPIRA-RULE-005` | Each request performs a fresh selection attempt. | For each `GET /` request `r`, the shell shall read a quote file snapshot and request one random index for that request. | `capability` | `capability` | `QuoteFileReader.read_text()` and `RandomIndexSource.choose_index(n)` used once per request | `none` | request 1 uses index `2`, request 2 uses index `0` | shell reuses previous request's selected quote without calling the random source | Implementation is non-conformant. |
| `INSPIRA-RULE-006` | A fresh selection does not guarantee a different displayed quote. | If two requests receive the same valid `QuoteIndex`, they may display the same `QuoteText`. | `representation` | `representation` | No "previous quote" state exists in the domain model. | `none` | indexes `0`, `0` on catalog `["A", "B"]` display `"A"` twice | any implementation that rejects same-as-previous results solely because they repeat | Repetition suppression is not permitted unless the requirement changes. |
| `INSPIRA-RULE-007` | Quote content is rendered as text, not executable markup. | `render_page(q)` shall HTML-escape `q` before interpolation into the document. | `operation` | `operation` | Pure renderer `render_page(QuoteText) -> QuotePage` | `none` | quote `<script>alert("x")</script>` renders as escaped text | raw `<script>` appears unescaped in the response body | Return is non-conformant because it exposes script markup. |
| `INSPIRA-RULE-008` | Configuration failures surface as temporary unavailability. | If file read fails or catalog construction fails, then response = HTTP 503 with body containing `"Quotes are temporarily unavailable."` and no quote. | `operation` | `operation` | `CatalogError -> ErrorPageResponse` mapping | `none` | `file_missing`, `catalog_empty` | returning HTTP 200 with blank quote area | Shell returns 503 error page. |
| `INSPIRA-RULE-009` | File changes affect the next refresh. | For requests `r1`, `r2` where the quote file changes between them, `r2` uses a fresh file snapshot, not a cached pre-change snapshot. | `capability` | `capability` | Per-request `QuoteFileReader.read_text()` | `none` | request 1 sees catalog `["A"]`, request 2 sees catalog `["A", "B"]` | request 2 still selects only from stale catalog `["A"]` | Implementation is non-conformant. |
| `INSPIRA-RULE-010` | The pure core has no direct filesystem or RNG access. | Core functions accept raw text, typed catalogs, and typed indexes; they do not call filesystem or randomness APIs. | `capability` | `capability` | Effect-free core module plus injected `Protocol` ports in the shell | `none` | `parse_catalog(raw_text)`, `select_quote(catalog, index)`, `render_page(quote)` | core function calling `open()` or `random.randrange()` | Implementation is non-conformant. |

## 5. Functional Core / Imperative Shell Split

### Pure Core Responsibilities

- construct `QuoteText` values from raw lines
- construct `QuoteCatalog` from a raw text snapshot
- validate non-empty and duplicate-free catalogs
- select a `QuoteText` from a valid catalog given a valid `QuoteIndex`
- render a `QuotePage` from `QuoteText` with HTML escaping
- map typed `CatalogError` values to deterministic error-page payloads

### Imperative Shell Responsibilities

- receive `GET /` requests
- read the configured quote file from disk
- translate filesystem failures into typed `CatalogError` values
- request one random index per HTTP request
- transform the selected or error page payload into an HTTP response

### Ports And Capabilities

| Name | Kind | Used By | Purpose | Allowed Operations |
|------|------|---------|---------|--------------------|
| `QuoteFileReader` | `capability` | shell | Read the configured quote file as UTF-8 text. | `read_text(path) -> str` |
| `RandomIndexSource` | `protocol` | shell | Supply one random index in `[0, n)` for a given catalog size. | `choose_index(upper_bound_exclusive: int) -> int` |
| `HttpResponseFactory` | `adapter` | shell | Convert pure page payloads into framework-specific HTTP responses. | `ok_html(body)`, `service_unavailable_html(body)` |

### Determinism And Injected Dependencies

| Dependency | Why It Must Be Injected | Injection Shape |
|------------|--------------------------|-----------------|
| quote file contents | Keeps the core effect-free and lets tests provide snapshots directly. | `QuoteFileReader` used only in the shell |
| random index source | Makes quote selection reproducible in tests and keeps randomness out of the core. | `RandomIndexSource` protocol |
| HTTP framework response object | Keeps the renderer framework-agnostic and deterministic. | `HttpResponseFactory` adapter |

## 6. Error Taxonomy

| Error ID | Trigger | Detection Layer | User/Operator Surface | Retryable | Notes |
|----------|---------|-----------------|-----------------------|-----------|-------|
| `CATALOG_FILE_MISSING` | Configured quote file path does not exist. | shell capability | User sees HTTP 503 with `"Quotes are temporarily unavailable."`; operator log includes the path. | `yes` | Becomes successful once the file is restored. |
| `CATALOG_FILE_UNREADABLE` | File exists but cannot be opened or decoded as UTF-8. | shell capability | User sees HTTP 503; operator log includes the underlying read/decode failure. | `yes` | Covers permission and decoding failures. |
| `CATALOG_EMPTY` | After trimming and blank-line removal, no quotes remain. | core construction | User sees HTTP 503; operator log identifies an empty catalog. | `yes` | This is a configuration error, not a user-input error. |
| `CATALOG_DUPLICATE_QUOTE` | Two or more trimmed quote lines are identical. | core construction | User sees HTTP 503; operator log identifies duplicate quote text. | `yes` | Prevents duplicate text from biasing selection probability. |
| `RANDOM_INDEX_OUT_OF_RANGE` | Shell supplies an index outside `[0, len(catalog))`. | shell/core boundary | User sees HTTP 503; operator log identifies a broken random source contract. | `yes` | Indicates implementation or adapter defect. |

## 7. Non-Functional Constraints

| Constraint | Requirement |
|------------|-------------|
| Idempotency | `GET /` shall not mutate server-side domain state. |
| Ordering | `N/A — none, because requests are independent and no cross-request ordering semantics are required.` |
| Concurrency | Concurrent requests may be served independently; no request may require knowledge of the previously served quote. |
| Performance | For a quote file of up to 1,000 non-empty lines on local disk, the server should return a response within 250 ms at the 95th percentile under normal single-node operation. |
| Durability | `N/A — none, because the application does not persist server-side domain state.` |
| Observability | Each 503 response shall log the `CatalogError` identifier and configured file path. Successful requests may be logged without quote content. |
| Security | Quote text shall be HTML-escaped before rendering, and the quote file path shall come from server configuration, not user input. |
| Resource ceilings | The quote file shall be treated as a small local text asset; catalogs larger than 1,000 non-empty lines are out of scope for this toy example. |

## 8. External Interfaces And Assume/Guarantee Contracts

### HTTP `GET /`

- Requires: The request uses method `GET`; no query parameters or request body are required.
- Guarantees: On success, returns HTTP 200 `text/html` containing exactly one displayed quote.
- Fails with: `CATALOG_FILE_MISSING`, `CATALOG_FILE_UNREADABLE`, `CATALOG_EMPTY`,
  `CATALOG_DUPLICATE_QUOTE`, `RANDOM_INDEX_OUT_OF_RANGE`.
- Preserves: No server-side domain state is mutated by serving the page.

### Quote File

- Requires: A configured UTF-8 text file exists at the server-defined path; each non-empty line is
  intended to represent one quote.
- Guarantees: The next `GET /` request uses the latest file contents visible at request time.
- Fails with: `CATALOG_FILE_MISSING`, `CATALOG_FILE_UNREADABLE`, `CATALOG_EMPTY`,
  `CATALOG_DUPLICATE_QUOTE`.
- Preserves: The core receives an immutable raw-text snapshot for each request.

### Random Index Source

- Requires: When asked for `n`, returns one integer in the half-open range `[0, n)`.
- Guarantees: One selection attempt is made per request.
- Fails with: `RANDOM_INDEX_OUT_OF_RANGE`.
- Preserves: The core remains free of direct randomness APIs.

## 9. Non-Goals

- guarantee a different quote from the previous refresh
- render quote authors, categories, or images
- support multiple pages, APIs, or administrative workflows
- persist browsing history or analytics
- fetch quotes from a database or external service

## 10. Glossary And Assumptions

### Glossary

| Term | Meaning |
|------|---------|
| `refresh` | A new browser request to `GET /`, whether via reload, revisit, or direct navigation. |
| `fresh selection attempt` | A new invocation of the random index source for the current request. |
| `quote file snapshot` | The exact UTF-8 text contents read from the configured quote file for one request. |
| `configuration error page` | The HTTP 503 HTML response containing `"Quotes are temporarily unavailable."` and no quote. |

### Assumptions

| Assumption | Why It Is Safe | What Changes If False |
|------------|----------------|-----------------------|
| `"A new quote displays each time the page is refreshed"` means a fresh random draw, not a uniqueness guarantee. | This is the least stateful interpretation and fits the functional-core / imperative-shell preference. | If uniqueness across refreshes is required, the spec needs request history or client/session state and a new rule set. |
| The quote file is curated by an operator rather than by end users. | It keeps write workflows, moderation, and authentication out of scope for this toy example. | If end users can edit quotes, new requirements are needed for validation, authorization, and persistence. |
| Per-request file reads are acceptable for a small local text file. | It keeps update visibility simple and avoids cache invalidation rules in the toy example. | If the file is large or hosted remotely, caching and refresh semantics need to be specified. |
