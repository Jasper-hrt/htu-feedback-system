# TODO — Mobile UI Layout Fixes

Task: Fix mobile UI overlaps at the top (header / Public Board / SRC Admin / profile / Dark Mode / Welcome) and prevent feature cards & bottom nav from being cut off or covering content. Only edit `static/css/mobile.css`. No design/color/functionality changes.

## Steps

- [x] 0. Explore repo (base.html, style.css, mobile.css, main.js) to understand the overlap root cause.
- [x] 1. Get user approval for the edit plan.
- [ ] 2. Hide desktop nav links inside `.navbar-links` on mobile (keep only theme toggle) — removes header overlap.
- [ ] 3. Reinforce bottom-nav clearance so the floating nav doesn't cover page content.
- [ ] 4. Prevent feature/hero cards from being clipped (no horizontal overflow; full-width stacking).
- [ ] 5. Verify the changes (CSS syntax check / review).
