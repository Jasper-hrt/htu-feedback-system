# TODO — Mobile App-Style Redesign

## Goal
Transform the responsive website into a native-app feel on mobile widths (≤768px) across all pages, without touching the desktop layout.

## Steps
- [x] **1. Hide desktop footer on mobile** — the floating bottom nav replaces it (app-like).
- [x] **2. Tighten main container** — full-width, reduced padding on small screens.
- [x] **3. App-style sticky header** — slim translucent bar, hide inline theme toggle text, subtle border.
- [x] **4. Upgrade bottom nav** — "lifted" active pill, animated icon bump, badge dots for forum/chat/news.
- [x] **5. Page transition** — fade/slide on `.main-container` across routes.
- [x] **6. Compact app-style hero banners** — reduced padding, left-aligned, full-width.
- [x] **7. Mobile polish** — hide `.footer-brand`, tighter cards/tables, safe-area handling, disable hover zoom.

## Badge Fixes (this session)
- Registered `initMobileBadges()` in the `DOMContentLoaded` handler in `static/js/main.js` (it existed but was never called).
- Added `.bnav-badge` styling to `static/css/mobile.css` (position, error-red gradient, pop animation, dark-mode shadow) so the unread badges on the bottom nav actually render.
- Verified `base.html` badge markup (`<span class="bnav-badge" data-badge="...">`) matches the JS/CSS selectors.
- `node --check` passes on `main.js`.

## Dependent Files
- `static/css/mobile.css` (primary)
- `templates/base.html` (badge dots on bottom nav)
- `static/js/main.js` (sync active badge on route)

## Follow-up
- Verify no desktop regression.
- Optionally run Flask to preview at mobile width.
