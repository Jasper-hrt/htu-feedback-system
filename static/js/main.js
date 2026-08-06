// HTU SRC Feedback System - Main JavaScript

function initUrgencySlider() {
    const slider = document.getElementById('urgencySlider');
    const urgencySpan = document.getElementById('urgencyValue');
    if (!slider) return;
    
    slider.addEventListener('input', function() {
        urgencySpan.textContent = this.value;
    });
    
    const textarea = document.getElementById('feedbackText');
    if (textarea) {
        const urgencyKeywords = {
            5: ['emergency', 'danger', 'injured', 'unsafe', 'hazard', 'assault', 'harassment'],
            4: ['weeks', 'ignored', 'still', 'no water', 'no power', 'broken', 'flood'],
            3: ['delay', 'late', 'rude', 'unhelpful', 'expensive'],
            2: ['slow', 'small', 'noisy'],
            1: ['suggestion', 'maybe']
        };
        
        textarea.addEventListener('input', function() {
            const text = this.value.toLowerCase();
            let detectedUrgency = 1;
            for (let urgency = 5; urgency >= 1; urgency--) {
                for (const keyword of urgencyKeywords[urgency]) {
                    if (text.includes(keyword)) {
                        detectedUrgency = urgency;
                        break;
                    }
                }
                if (detectedUrgency > 1) break;
            }
            if (!slider.dataset.manuallyChanged) {
                slider.value = detectedUrgency;
                urgencySpan.textContent = detectedUrgency;
            }
        });
        
        slider.addEventListener('mousedown', function() {
            slider.dataset.manuallyChanged = 'true';
        });
    }
}

function initFollowupToggle() {
    const checkbox = document.getElementById('followupCheckbox');
    const div = document.getElementById('followupEmailDiv');
    if (checkbox && div) {
        checkbox.addEventListener('change', function() {
            div.classList.toggle('hidden', !this.checked);
        });
    }
}

function initAdminFilters() {
    const searchInput = document.getElementById('searchInput');
    const categoryFilter = document.getElementById('categoryFilter');
    const statusFilter = document.getElementById('statusFilter');
    const urgencyFilter = document.getElementById('urgencyFilter');
    
    if (!searchInput) return;
    
    function filterTable() {
        const search = searchInput.value.toLowerCase();
        const category = categoryFilter?.value || '';
        const status = statusFilter?.value || '';
        const urgency = urgencyFilter?.value || '';
        const rows = document.querySelectorAll('#feedbackTableBody tr');
        
        rows.forEach(row => {
            const text = row.cells[1]?.innerText.toLowerCase() || '';
            const rowCategory = row.cells[2]?.innerText || '';
            const rowStatus = row.cells[5]?.innerText || '';
            const urgencyScore = parseInt(row.cells[4]?.innerText.match(/\d+/)?.[0] || '0');
            
            let match = text.includes(search);
            if (category && rowCategory !== category) match = false;
            if (status && rowStatus !== status) match = false;
            if (urgency === '4' && urgencyScore < 4) match = false;
            if (urgency === '5' && urgencyScore < 5) match = false;
            row.style.display = match ? '' : 'none';
        });
    }
    
    searchInput.addEventListener('keyup', filterTable);
    if (categoryFilter) categoryFilter.addEventListener('change', filterTable);
    if (statusFilter) statusFilter.addEventListener('change', filterTable);
    if (urgencyFilter) urgencyFilter.addEventListener('change', filterTable);
}

function showResponseModal(id, currentResponse) {
    const modal = document.getElementById('responseModal');
    const form = document.getElementById('responseForm');
    if (modal && form) {
        form.action = `/admin/update/${id}`;
        const textarea = form.querySelector('textarea');
        if (textarea) textarea.value = currentResponse;
        modal.classList.add('active');
    }
}

function closeModal() {
    const modal = document.getElementById('responseModal');
    if (modal) modal.classList.remove('active');
}

document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') closeModal();
});

function filterStatus(status) {
    const cards = document.querySelectorAll('.feedback-card');
    let visibleCount = 0;
    
    cards.forEach(card => {
        const cardStatus = card.getAttribute('data-status');
        if (status === 'all' || cardStatus === status) {
            card.style.display = 'block';
            visibleCount++;
        } else {
            card.style.display = 'none';
        }
    });
    
    const buttons = document.querySelectorAll('.filter-btn');
    buttons.forEach(btn => {
        btn.style.opacity = '0.7';
        btn.style.transform = 'scale(1)';
    });
    
    const activeButton = document.getElementById(`filter${status === 'all' ? 'All' : status.replace(' ', '')}`);
    if (activeButton) {
        activeButton.style.opacity = '1';
        activeButton.style.transform = 'scale(1.02)';
    }
    
    let noResultsMsg = document.getElementById('noResultsMsg');
    if (!noResultsMsg && visibleCount === 0 && status !== 'all') {
        noResultsMsg = document.createElement('div');
        noResultsMsg.id = 'noResultsMsg';
        noResultsMsg.className = 'card text-center';
        noResultsMsg.style.padding = '2rem';
        noResultsMsg.innerHTML = `<p style="color: var(--secondary);">No feedback with this status.</p><a href="/submit" class="btn btn-primary" style="margin-top: 1rem;">Submit New Feedback</a>`;
        document.getElementById('feedbackContainer')?.appendChild(noResultsMsg);
    } else if (noResultsMsg) {
        noResultsMsg.style.display = visibleCount === 0 && status !== 'all' ? 'block' : 'none';
    }
}

function voteFeedback(feedbackId) {
    fetch(`/vote/${feedbackId}`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                const countSpan = document.getElementById(`vote-count-${feedbackId}`);
                if (countSpan) countSpan.textContent = data.vote_count;
            }
        })
        .catch(err => console.error('Vote failed:', err));
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text);
    alert('Template copied to clipboard!');
}



function initThemeToggle() {
    const toggle = document.getElementById('themeToggle');
    if (!toggle) return;


    const saved = localStorage.getItem('theme'); // 'light' | 'dark'
    const systemPrefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const isDark = saved ? saved === 'dark' : systemPrefersDark;

    document.documentElement.classList.toggle('theme-dark', isDark);
    toggle.checked = isDark;

    const themeUi = document.querySelector('.theme-toggle-ui');

    function updateUI(dark) {
        if (!themeUi) return;
        themeUi.textContent = dark ? '🌙 Dark' : '☀️ Light';
    }

    updateUI(isDark);


    toggle.addEventListener('change', function() {
        const dark = this.checked;
        document.documentElement.classList.toggle('theme-dark', dark);
        localStorage.setItem('theme', dark ? 'dark' : 'light');
        updateUI(dark);
    });



}



// ============================================
// 🍔 Navbar Mobile Menu Toggle
// ============================================
function initNavbarMobileMenu() {
    const hamburger = document.getElementById('navbarHamburger');
    const menu = document.getElementById('navbarMobileMenu');
    if (!hamburger || !menu) return;

    function setMenu(open) {
        menu.classList.toggle('open', open);
        hamburger.classList.toggle('active', open);
        hamburger.setAttribute('aria-expanded', open ? 'true' : 'false');
    }

    hamburger.addEventListener('click', function(e) {
        e.stopPropagation();
        setMenu(!menu.classList.contains('open'));
    });

    // Close when a link inside the mobile menu is clicked
    menu.querySelectorAll('a').forEach(function(link) {
        link.addEventListener('click', function() {
            setMenu(false);
        });
    });

    // Close on outside click
    document.addEventListener('click', function(e) {
        if (menu.classList.contains('open') &&
            !menu.contains(e.target) &&
            !hamburger.contains(e.target)) {
            setMenu(false);
        }
    });

    // Close on Escape
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && menu.classList.contains('open')) {
            setMenu(false);
        }
    });

// Sync mobile theme toggle with the main theme toggle
    const mainToggle = document.getElementById('themeToggle');
    const mobileToggle = document.getElementById('themeToggleMobile');
    if (mainToggle && mobileToggle) {
        // Sync the mobile toggle's initial state with the main toggle on load
        mobileToggle.checked = mainToggle.checked;
        const mobileThemeUi = mobileToggle.closest('.theme-toggle')?.querySelector('.theme-toggle-ui');
        if (mobileThemeUi) mobileThemeUi.textContent = mainToggle.checked ? '🌙 Dark' : '☀️ Light';

        // Keep both checkboxes in sync
        mainToggle.addEventListener('change', function() {
            mobileToggle.checked = mainToggle.checked;
            if (mobileThemeUi) mobileThemeUi.textContent = mainToggle.checked ? '🌙 Dark' : '☀️ Light';
        });
        mobileToggle.addEventListener('change', function() {
            mainToggle.checked = mobileToggle.checked;
            // Trigger the theme change
            const dark = this.checked;
            document.documentElement.classList.toggle('theme-dark', dark);
            localStorage.setItem('theme', dark ? 'dark' : 'light');
            const themeUi = document.querySelector('.theme-toggle-ui');
            if (themeUi) themeUi.textContent = dark ? '🌙 Dark' : '☀️ Light';
            if (mobileThemeUi && mobileThemeUi !== themeUi) mobileThemeUi.textContent = dark ? '🌙 Dark' : '☀️ Light';
        });
    }
}

// ============================================
// 🎬 Scroll Reveal Animations
// ============================================
function initScrollReveal() {
    const revealElements = document.querySelectorAll('.reveal, .reveal-left, .reveal-right, .reveal-scale');
    
    if (revealElements.length === 0) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });
    
    revealElements.forEach(el => observer.observe(el));
}

// ============================================
// 🔢 Animated Counters
// ============================================
function initCounters() {
    const counters = document.querySelectorAll('.counter');
    if (counters.length === 0) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const counter = entry.target;
                const target = parseInt(counter.dataset.target) || 0;
                animateCounter(counter, target);
                observer.unobserve(counter);
            }
        });
    }, { threshold: 0.5 });
    
    counters.forEach(counter => observer.observe(counter));
}

function animateCounter(element, target) {
    const duration = 2000;
    const steps = 60;
    const increment = target / steps;
    let current = 0;
    let step = 0;
    
    const timer = setInterval(() => {
        step++;
        current = Math.min(Math.round(increment * step), target);
        element.textContent = current;
        
        if (current >= target) {
            clearInterval(timer);
            element.textContent = target;
        }
    }, duration / steps);
}

// ============================================
// 📢 Announcements Search/Filter
// ============================================
function initAnnouncementFilters() {
    const searchInput = document.getElementById('announcementSearch');
    const filterPills = document.querySelectorAll('.announcements-filter-pill');
    if (!searchInput && filterPills.length === 0) return;
    
    function filterAnnouncements() {
        const search = searchInput ? searchInput.value.toLowerCase() : '';
        const activePill = document.querySelector('.announcements-filter-pill.active');
        const filter = activePill ? activePill.dataset.filter : 'all';
        const cards = document.querySelectorAll('.announcement-card-v2');
        const noResults = document.querySelector('.announcements-no-results');
        let visibleCount = 0;
        
        cards.forEach(card => {
            const title = (card.dataset.title || '').toLowerCase();
            const content = (card.dataset.content || '').toLowerCase();
            const category = card.dataset.category || '';
            
            const matchesSearch = title.includes(search) || content.includes(search);
            const matchesFilter = filter === 'all' || category === filter;
            
            card.style.display = (matchesSearch && matchesFilter) ? '' : 'none';
            if (matchesSearch && matchesFilter) visibleCount++;
        });
        
        if (noResults) {
            noResults.classList.toggle('visible', visibleCount === 0);
        }
    }
    
    if (searchInput) {
        searchInput.addEventListener('input', filterAnnouncements);
    }
    
    filterPills.forEach(pill => {
        pill.addEventListener('click', function() {
            filterPills.forEach(p => p.classList.remove('active'));
            this.classList.add('active');
            filterAnnouncements();
        });
    });
}

function toggleAnnouncement(index) {
    const body = document.getElementById('announcementBody' + index);
    if (!body) return;
    const btn = body.parentElement.querySelector('.announcement-card-toggle-v2');
    if (!btn) return;
    
    if (body.classList.contains('collapsed')) {
        body.classList.remove('collapsed');
        btn.innerHTML = '🔼 Show less';
    } else {
        body.classList.add('collapsed');
        btn.innerHTML = '📖 Read more';
    }
}

// ============================================
// 🔐 Password Visibility Toggle
// ============================================
function togglePassword(inputId, button) {
    const input = document.getElementById(inputId);
    if (!input) return;
    if (input.type === 'password') {
        input.type = 'text';
        button.textContent = '🙈';
    } else {
        input.type = 'password';
        button.textContent = '👁️';
    }
}

// ============================================
// 🖼️ Hero Parallax Effect
// ============================================
function initHeroParallax() {
    const heroBg = document.querySelector('.hero-bg-image img');
    if (!heroBg) return;
    
    window.addEventListener('scroll', function() {
        const scrollY = window.scrollY;
        const speed = 0.15;
        heroBg.style.transform = `scale(1.02) translateY(${scrollY * speed}px)`;
    }, { passive: true });
}

// ============================================
// 📸 Image Lazy Loading Enhancement
// ============================================
function initImageHandling() {
    // Fade in images on load
    document.querySelectorAll('.htu-logo-auth, .htu-logo-nav, .htu-logo-footer, .hero-bg-image img, .auth-bg-image img').forEach(img => {
        if (img.complete) {
            img.style.opacity = '1';
        } else {
            img.addEventListener('load', function() {
                this.style.opacity = '1';
            });
        }
        // Set initial opacity for smooth transition
        img.style.transition = 'opacity 0.4s ease';
        img.style.opacity = img.complete ? '1' : '0.3';
    });
    
    // Handle hero image error fallback
    document.querySelectorAll('.hero-bg-image img, .auth-bg-image img').forEach(img => {
        img.addEventListener('error', function() {
            this.style.display = 'none';
            const parent = this.closest('.hero-bg-image, .auth-bg-image');
            if (parent) {
                parent.style.display = 'none';
            }
        });
    });
}

// ============================================
// 🎯 Student Dropdown Menu
// ============================================
function initStudentDropdown() {
    const toggle = document.getElementById('studentDropdownToggle');
    const menu = document.getElementById('studentDropdownMenu');
    const backdrop = document.getElementById('studentDropdownBackdrop');
    
    if (!toggle || !menu) return;
    
    function openDropdown() {
        menu.classList.add('active');
        toggle.classList.add('active');
        if (backdrop) backdrop.classList.add('active');
    }
    
    function closeDropdown() {
        menu.classList.remove('active');
        toggle.classList.remove('active');
        if (backdrop) backdrop.classList.remove('active');
    }
    
    function toggleDropdown(e) {
        e.stopPropagation();
        if (menu.classList.contains('active')) {
            closeDropdown();
        } else {
            openDropdown();
        }
    }
    
    toggle.addEventListener('click', toggleDropdown);
    
    // Close on backdrop click
    if (backdrop) {
        backdrop.addEventListener('click', closeDropdown);
    }
    
    // Close on Escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && menu.classList.contains('active')) {
            closeDropdown();
        }
    });
    
    // Close on window resize (if small screen and menu is fixed)
    window.addEventListener('resize', function() {
        if (menu.classList.contains('active') && window.innerWidth <= 480) {
            // Keep open on mobile - it's fine
        }
    });
}

// ============================================
// 🎯 Active Page Highlight for Dropdown & Side Menu Links
// ============================================
function initActivePageHighlight() {
    const links = document.querySelectorAll('.student-dropdown-link[data-page], .student-side-menu-link[data-page]');
    if (links.length === 0) return;

    const path = window.location.pathname;
    let activePage = '';

    // Map URL path to a data-page value
    if (path.startsWith('/student/dashboard')) {
        activePage = 'dashboard';
    } else if (path.startsWith('/submit')) {
        activePage = 'submit';
    } else if (path.startsWith('/forum')) {
        activePage = 'forum';
    } else if (path.startsWith('/chat')) {
        activePage = 'chat';
    } else if (path.startsWith('/announcements')) {
        activePage = 'announcements';
    } else if (path.startsWith('/public')) {
        activePage = 'public';
    }

    if (!activePage) return;

    links.forEach(link => {
        if (link.dataset.page === activePage) {
            link.classList.add('active');
        }
    });
}

// ============================================
// 🧭 Student Side Menu Collapse Toggle
// ============================================
function initStudentSideMenu() {
    const menu = document.getElementById('studentSideMenu');
    const collapseBtn = document.getElementById('sideMenuToggle');
    if (!menu || !collapseBtn) return;

    const STORAGE_KEY = 'htu_student_side_menu_collapsed';

    // Restore persisted state
    if (localStorage.getItem(STORAGE_KEY) === '1') {
        menu.classList.add('collapsed');
    }

    collapseBtn.addEventListener('click', function() {
        const collapsed = menu.classList.toggle('collapsed');
        localStorage.setItem(STORAGE_KEY, collapsed ? '1' : '0');
    });
}

document.addEventListener('DOMContentLoaded', function() {
    initUrgencySlider();
    initFollowupToggle();
    initAdminFilters();
    initThemeToggle();
    initNavbarMobileMenu();
    initScrollReveal();
    initCounters();
    initAnnouncementFilters();
    initHeroParallax();
    initImageHandling();
    initStudentDropdown();
    initStudentSideMenu();
    initActivePageHighlight();
});



