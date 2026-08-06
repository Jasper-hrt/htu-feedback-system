# Mobile Responsiveness Fixes — HTU SRC System

## Tasks

- [x] A. Add mobile hamburger menu to navbar
  - [x] base.html: Add hamburger toggle button + collapsible mobile menu
  - [x] style.css: Add hamburger button + mobile menu styles
  - [x] main.js: Add toggle logic + close on outside click / link click + mobile theme toggle sync (initial state + both ways)
- [x] B. Make admin sidebar mobile-friendly (horizontal scroll at ≤992px)
- [x] C. Fix inline grid overrides that break mobile (admin_analytics 5-col stats grid) — added `!important` to responsive `.stats-grid` rules
- [x] D. General mobile polish (spacing, fonts, grid collapse on small screens) — CSS present

## Mobile-Only UI Redesign (new)

- [x] E. Create dedicated `static/css/mobile.css` (mobile-only, desktop untouched)
  - [x] App-style sticky bottom navigation bar (Home / Forum / Chat / News / Board)
  - [x] Compact mobile header (smaller logo/title, tighter hamburger)
  - [x] Touch-optimized inputs & buttons (16px font prevents iOS zoom)
  - [x] Student side-menu → horizontal chip bar on mobile
  - [x] Admin side-menu → horizontal chip bar on mobile
  - [x] Tables → stacked mobile cards (with `data-label` cell labels)
  - [x] Hero & section stacking (single-column, full-width actions)
  - [x] Chat full-screen messaging-app layout
  - [x] Forum & announcements mobile polish
  - [x] Auth pages mobile polish
  - [x] Floating Submit action button (FAB)
  - [x] Safe-area inset handling for notched phones
  - [x] Dark-mode compatible overrides
- [x] F. Link `mobile.css` in `base.html`
- [x] G. Add bottom-nav markup + FAB in `base.html`
- [x] H. Add `initMobileBottomNav()` + `initMobileFab()` in `main.js`
