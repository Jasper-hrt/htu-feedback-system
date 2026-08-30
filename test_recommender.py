"""
Comprehensive test suite for the new recommendation engine.
Tests category classification, sentiment analysis, student recommendations,
admin action plans, multi-issue handling, and fallback behavior.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommender import (
    generate_recommendation,
    classify_categories,
    analyze_sentiment_type,
    determine_urgency,
    CATEGORY_DEFINITIONS,
)
from sentiment_analyzer import process_feedback


# ==================== TEST DATA ====================

# Category-specific test cases (at least 5 per category)
CATEGORY_TEST_CASES = {
    "Academics": [
        "My lecturer for MATH 101 has not come to class for three weeks now",
        "The exam results for last semester have still not been released",
        "Our professor is very knowledgeable and explains concepts clearly",
        "The assignment I submitted was never marked or graded",
        "The syllabus for this course has not been covered at all",
        "My grades are always unfair and the marking is biased",
        "The lecturer speaks too fast and we cannot hear anything",
        "There was no tutorial for the whole semester",
        "I recommend that the school should provide more handouts",
        "The teaching is excellent and very engaging",
    ],
    "ICT": [
        "The wifi in the hostel has been down for two days",
        "I cannot login to the student portal, it says invalid credentials",
        "The computers in the lab are very slow and keep crashing",
        "The internet connection is too slow to download lecture materials",
        "The portal is down and I cannot register for courses",
        "The printer in the library is not working",
        "My student email is not receiving messages",
        "The network in the lecture hall is always disconnecting",
        "The website is not loading on my phone",
        "The new wifi system is very fast and reliable",
    ],
    "Finance": [
        "My school fees are too high and I cannot afford to pay",
        "I applied for a scholarship but have not received any response",
        "The finance office overcharged me on my tuition",
        "I have been waiting for my refund for over a month",
        "The payment I made has not been reflected in my account",
        "I cannot get a loan to pay my fees",
        "The bursary application process is too complicated",
        "I was charged extra fees that were not in the original invoice",
        "The mobile money payment option is not working",
        "The finance office staff are very helpful and efficient",
    ],
    "Safety": [
        "My phone was stolen from my hostel room last night",
        "There was a shooting near the campus gate yesterday",
        "I was attacked on my way back from the library at night",
        "The security guards are never at their post",
        "There is no lighting on the path to the hostel",
        "Someone broke into my room and took my laptop",
        "I feel unsafe walking alone on campus at night",
        "A student was kidnapped near the campus",
        "The CCTV cameras in the parking lot are not working",
        "I was harassed by a group of boys near the canteen",
    ],
    "Maintenance": [
        "The ceiling in my room has been leaking for a week",
        "The door to my hostel room is broken and does not lock properly",
        "There is no water flowing in the bathroom",
        "The fan in the lecture hall is not working",
        "The elevator in the science building is stuck",
        "The pipes in the kitchen have burst and water is everywhere",
        "The lights in the corridor are not working",
        "The window in my room is broken and cannot close",
        "The drainage system is blocked and causing flooding",
        "The roof of the library is leaking when it rains",
    ],
    "Accommodation": [
        "My hostel room is too small and overcrowded",
        "The bathroom in our hall is always dirty and smelly",
        "I have not been allocated a room for this semester",
        "My roommate is too noisy and disturbs my studies",
        "There are cockroaches everywhere in the hostel",
        "The warden is never available when we need help",
        "I want to change my room but the hall master is not cooperating",
        "The mattress on my bed is very uncomfortable",
        "There is no hot water for showering",
        "The hostel is too far from the lecture halls",
    ],
    "Staff": [
        "The lecturer is very rude to students who ask questions",
        "The secretary at the registry is always insulting students",
        "My professor never comes to class and does not care",
        "The security guard at the gate is very disrespectful",
        "The finance officer demanded a bribe before processing my payment",
        "The caterer is always shouting at students",
        "The librarian is very helpful and always assists us",
        "The lecturer makes inappropriate comments to female students",
        "The cleaner never cleans the offices",
        "The department head ignores all our complaints",
    ],
    "Administration": [
        "I have been waiting for my transcript for over two months",
        "The admission process is too slow and bureaucratic",
        "The registry office is always closed when I go there",
        "My certificate is still not ready after graduation",
        "The registration process is too complicated",
        "I submitted my forms but have not received any response",
        "The front desk staff are very unhelpful",
        "I need a verification letter but no one is helping me",
        "The office takes too long to process simple requests",
        "The new online registration system is very efficient",
    ],
    "Transport": [
        "The campus bus is always late and sometimes does not come",
        "There is no bus service to the main campus",
        "The bus driver was driving recklessly and speeding",
        "The parking lot is always full and I cannot find space",
        "The transport fare has increased without notice",
        "The bus broke down on the way to campus",
        "There are not enough buses for all the students",
        "The bus schedule is not convenient for evening classes",
        "The road to the campus is full of potholes",
        "The new shuttle service is very reliable and comfortable",
    ],
    "Catering": [
        "The food in the canteen is always cold and tasteless",
        "I found a worm in my rice at the dining hall",
        "The food prices are too high for students to afford",
        "The kitchen is very dirty and unhygienic",
        "The portions are too small and I am still hungry",
        "The same food is served every day with no variety",
        "I got food poisoning after eating at the canteen",
        "The canteen staff are very rude to students",
        "There are flies all over the food in the kitchen",
        "The food is delicious and well prepared",
    ],
    "Library": [
        "The library is always full and there are no seats available",
        "I cannot find the books I need for my research",
        "The library closes too early in the evening",
        "The e-library is not working and I cannot access journals",
        "The librarian is never at the help desk",
        "The computers in the library are very slow",
        "The library is too noisy for studying",
        "I need a book that the library does not have",
        "The air conditioning in the library is not working",
        "The library has excellent resources and a great study environment",
    ],
    "Student Affairs": [
        "The health center is always closed when students need help",
        "There are no sports facilities for students",
        "The counseling center has no counselor available",
        "My ID card has not been issued after three months",
        "The football field is in very poor condition",
        "There are no organized events for students",
        "The gym equipment is broken and outdated",
        "I need mental health support but there is no one to talk to",
        "The student clubs are not being supported by the school",
        "The orientation program was very well organized",
    ],
}

# Difficult edge cases
DIFFICULT_CASES = [
    # Multi-issue feedback
    {
        "text": "The wifi is not working and the food in the canteen is terrible",
        "expected_categories": ["ICT", "Catering"],
        "description": "Two distinct issues in one feedback",
    },
    {
        "text": "My lecturer is always absent and the classroom has no lights",
        "expected_categories": ["Academics", "Maintenance"],
        "description": "Academic + facilities issue",
    },
    {
        "text": "I was robbed near the hostel and the security guards were not around",
        "expected_categories": ["Safety", "Accommodation"],
        "description": "Safety + accommodation",
    },
    # Mixed sentiment
    {
        "text": "The food is good but the service is very slow",
        "expected_sentiment": "mixed",
        "description": "Positive + negative in one sentence",
    },
    {
        "text": "The lecturer is knowledgeable but always comes late",
        "expected_sentiment": "mixed",
        "description": "Positive quality + negative behavior",
    },
    {
        "text": "The hostel is clean but too noisy at night",
        "expected_sentiment": "mixed",
        "description": "Positive condition + negative environment",
    },
    # Resolution language
    {
        "text": "The wifi used to be very slow but it is much better now",
        "expected_sentiment": "mixed",
        "description": "Past problem, now resolved",
    },
    {
        "text": "There was a leak in my room but maintenance fixed it last week",
        "expected_sentiment": "mixed",
        "description": "Problem resolved by maintenance",
    },
    # Positive feedback
    {
        "text": "Thank you for the excellent service at the registry",
        "expected_sentiment": "positive",
        "description": "Simple positive feedback",
    },
    {
        "text": "The new library is amazing, very spacious and well-equipped",
        "expected_sentiment": "positive",
        "description": "Enthusiastic positive feedback",
    },
    # Unclear/very short
    {
        "text": "ok",
        "expected_sentiment": "unclear",
        "description": "Very short, unclear feedback",
    },
    {
        "text": "something is wrong",
        "expected_sentiment": "unclear",
        "description": "Vague feedback",
    },
    # Safety critical
    {
        "text": "There was a shooting near the campus last night",
        "expected_categories": ["Safety"],
        "expected_urgency": "critical",
        "description": "Critical safety incident",
    },
    {
        "text": "A student was kidnapped at the campus gate",
        "expected_categories": ["Safety"],
        "expected_urgency": "critical",
        "description": "Kidnapping incident",
    },
    # Ghanaian slang
    {
        "text": "The wifi no dey work at all in my hostel",
        "expected_categories": ["ICT"],
        "description": "Ghanaian slang for wifi not working",
    },
    {
        "text": "Chale, the food be bad paa yesterday",
        "expected_categories": ["Catering"],
        "description": "Ghanaian slang for food being very bad",
    },
    # Negation handling
    {
        "text": "There is no wifi in the lecture hall",
        "expected_categories": ["ICT"],
        "description": "Negation of wifi availability",
    },
    {
        "text": "The lecturer does not explain anything clearly",
        "expected_categories": ["Academics"],
        "description": "Negation of teaching quality",
    },
    # Sarcasm
    {
        "text": "The portal is great until you need to register",
        "expected_categories": ["ICT"],
        "description": "Sarcastic comment about portal",
    },
    {
        "text": "The bus is technically working, it just breaks down every day",
        "expected_categories": ["Transport"],
        "description": "Sarcastic comment about bus",
    },
]

# Additional challenging cases (100+)
ADDITIONAL_CASES = [
    # Academics edge cases
    "The lecturer for CS 201 is brilliant but the course material is outdated",
    "I have not received my exam results from last semester and it's been 3 months",
    "The professor cancelled class again without any notice",
    "Our assignment grades are not posted and the deadline for appeal has passed",
    "The teaching assistant is very helpful and always available for consultation",
    "The lab sessions are always cancelled due to lack of equipment",
    "The lecturer makes the subject very interesting and engaging",
    "I failed the course because the grading was completely unfair",
    "The course outline was never followed and we did not cover half the syllabus",
    "The lecturer is always available during office hours and very supportive",
    
    # ICT edge cases
    "The student portal crashes every time I try to check my results",
    "I cannot access my student email because I forgot my password",
    "The wifi in the library is excellent but in the hostel it's terrible",
    "The computer lab needs new keyboards, half of them are not working",
    "The new student app is very user-friendly and helpful",
    "The network is so slow that I cannot submit my assignment online",
    "The printer in the library has been out of paper for a week",
    "I was locked out of my account after entering the wrong password three times",
    "The website does not load properly on mobile devices",
    "The ICT helpdesk resolved my issue very quickly, thank you",
    
    # Finance edge cases
    "I paid my fees but the system still shows I have an outstanding balance",
    "The scholarship application deadline was extended, thank you",
    "I was charged for accommodation even though I am a day student",
    "The payment receipt was not generated after I paid online",
    "The bursary I was promised has not been paid for two semesters",
    "The finance office is very transparent about fee breakdown now",
    "I cannot afford the new fee increment, it is too much",
    "The refund for the cancelled course has not been processed",
    "The mobile money payment option makes it very convenient to pay",
    "I was overcharged on my tuition and need a refund",
    
    # Safety edge cases
    "Someone snatched my phone near the campus gate this morning",
    "The street lights on the path to the hostel are not working",
    "I saw a suspicious person lurking around the hostel last night",
    "The security at the gate does not check who comes in and out",
    "There was a fight between students near the canteen",
    "I feel very safe on campus now with the new security measures",
    "My bag was snatched by two boys on a motorcycle",
    "The CCTV cameras in the parking lot are just for show, they do not work",
    "A student was assaulted near the sports complex",
    "The security patrol has improved and I feel safer now",
    
    # Maintenance edge cases
    "The tap in the bathroom has been running non-stop for days",
    "The ceiling paint is peeling off and falling on our books",
    "The door lock is broken and anyone can enter my room",
    "The window cannot close properly and rain comes in",
    "The maintenance team fixed the plumbing issue very quickly",
    "The floor tiles in the hallway are cracked and dangerous",
    "The light switch in my room gives electric shocks",
    "The bathroom drain is blocked and water overflows",
    "The fan is making a loud noise and disturbing our sleep",
    "The new air conditioning system in the lecture hall is excellent",
    
    # Accommodation edge cases
    "I have been on the accommodation waiting list for two semesters",
    "The hostel room I was given is infested with rats",
    "My roommate plays music loudly at night and I cannot sleep",
    "The hall warden is very strict and does not allow visitors",
    "The new hostel block is very comfortable and well-maintained",
    "There is no space in the hostel and I have to commute long distances",
    "The bathroom is shared by too many people and is always dirty",
    "I was promised a single room but given a shared room",
    "The hostel has good security and I feel safe",
    "The kitchen in the hostel is not usable, nothing works",
    
    # Staff edge cases
    "The lecturer always insults students who ask questions",
    "The registry staff are very rude and unhelpful",
    "My lecturer is the best teacher I have ever had",
    "The security guard at the gate harasses students for no reason",
    "The finance officer is very professional and helpful",
    "The caterer always gives small portions and charges a lot",
    "The librarian goes out of her way to help students find resources",
    "The lecturer makes sexual comments that make female students uncomfortable",
    "The cleaner does not clean the classrooms regularly",
    "The department secretary is very efficient and responsive",
    
    # Administration edge cases
    "I have been coming to the registry for a week and still cannot get my documents",
    "The admission letter was sent to the wrong email address",
    "The new online transcript request system is very efficient",
    "I submitted my forms a month ago and have not heard anything",
    "The front desk staff are very welcoming and helpful",
    "The certificate collection process is too complicated and slow",
    "I need a reference letter urgently but no one is helping me",
    "The registration process was smooth and well-organized",
    "The office closes too early and students cannot get help after class",
    "The verification letter I requested has not been ready for weeks",
    
    # Transport edge cases
    "The bus did not show up and I missed my exam",
    "The driver was drunk and driving very recklessly",
    "There is no bus service on weekends",
    "The parking fee is too high for students",
    "The new shuttle route is very convenient for students",
    "The bus is always overcrowded and some students have to stand",
    "The road to the hostel is flooded whenever it rains",
    "The bus schedule does not match the class timetable",
    "I was charged twice for the same trip",
    "The transport office resolved my complaint very quickly",
    
    # Catering edge cases
    "I found a cockroach in my soup at the dining hall",
    "The food is always served cold and tastes bland",
    "The canteen is very clean and the food is hygienic",
    "The portions have become smaller but the price is the same",
    "I had stomach pains after eating at the canteen",
    "The menu needs more variety, we eat the same food every week",
    "The food vendor outside the campus is cheaper and better",
    "The water at the canteen is not clean, it has a strange taste",
    "The breakfast service starts too late for early morning classes",
    "The new caterer has improved the food quality significantly",
    
    # Library edge cases
    "The library is too noisy and I cannot concentrate",
    "I have been looking for this book for weeks and it is never available",
    "The e-library databases are very useful for research",
    "The library hours should be extended during exams",
    "The study rooms are always booked and unavailable",
    "The librarian helped me find resources I could not find on my own",
    "The library is too hot and there is no ventilation",
    "The photocopier is always broken or out of paper",
    "The new journal subscriptions are very helpful for my research",
    "The library Wi-Fi is faster than the rest of campus",
    
    # Student Affairs edge cases
    "The health center has no doctor available most of the time",
    "I have been trying to get my ID card for three months",
    "The counseling services have helped me a lot with my anxiety",
    "The sports equipment is old and dangerous to use",
    "The student clubs are not given any support or funding",
    "The orientation for freshmen was very informative and well-organized",
    "The gym is too small for the number of students",
    "I reported a case of bullying but nothing was done",
    "The cultural night was a great event, well done to the organizers",
    "The health center runs out of basic medicines frequently",
]


# ==================== TEST FUNCTIONS ====================

def test_category_classification():
    """Test that categories are correctly classified."""
    print("\n" + "="*70)
    print("TEST: Category Classification")
    print("="*70)
    
    passed = 0
    failed = 0
    total = 0
    
    for category, cases in CATEGORY_TEST_CASES.items():
        for case in cases:
            total += 1
            result = classify_categories(case)
            primary = result[0].name if result else "Other"
            
            if primary == category:
                passed += 1
            else:
                failed += 1
                print(f"  FAIL: '{case[:60]}...'")
                print(f"    Expected: {category}, Got: {primary}")
                print(f"    Evidence: {result[0].evidence[:3] if result else 'None'}")
    
    print(f"\n  Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    return passed, failed, total


def test_sentiment_analysis():
    """Test sentiment analysis for different types of feedback."""
    print("\n" + "="*70)
    print("TEST: Sentiment Analysis")
    print("="*70)
    
    test_cases = [
        # Positive cases
        ("The service is excellent and very helpful", "positive"),
        ("Thank you for the great work you are doing", "positive"),
        ("The new system is amazing and works perfectly", "positive"),
        ("I am very satisfied with the support I received", "positive"),
        ("The staff are professional and responsive", "positive"),
        
        # Negative cases
        ("The service is terrible and unhelpful", "negative"),
        ("I am very disappointed with the poor quality", "negative"),
        ("This is the worst experience I have ever had", "negative"),
        ("The system is broken and nobody cares", "negative"),
        ("I have been ignored and my problem is not solved", "negative"),
        
        # Mixed cases
        ("The food is good but the service is slow", "mixed"),
        ("The lecturer is knowledgeable but always late", "mixed"),
        ("The hostel is clean but too noisy", "mixed"),
        ("The wifi is fast but only in some areas", "mixed"),
        ("The system works well but crashes sometimes", "mixed"),
        
        # Neutral cases
        ("I would like to inquire about the registration process", "neutral"),
        ("The office is located near the main gate", "neutral"),
        ("I submitted my application last week", "neutral"),
        ("The meeting is scheduled for tomorrow", "neutral"),
        ("I need information about the course requirements", "neutral"),
    ]
    
    passed = 0
    failed = 0
    
    for text, expected in test_cases:
        analysis = process_feedback(text)
        sentiment_type = analyze_sentiment_type(text, analysis['sentiment'], analysis['sentiment_score'])
        
        if sentiment_type == expected:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: '{text[:60]}...'")
            print(f"    Expected: {expected}, Got: {sentiment_type}")
            print(f"    Sentiment: {analysis['sentiment']}, Score: {analysis['sentiment_score']}")
    
    total = passed + failed
    print(f"\n  Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    return passed, failed, total


def test_difficult_cases():
    """Test difficult edge cases."""
    print("\n" + "="*70)
    print("TEST: Difficult Edge Cases")
    print("="*70)
    
    passed = 0
    failed = 0
    
    for case in DIFFICULT_CASES:
        text = case["text"]
        result = generate_recommendation(text)
        
        case_passed = True
        errors = []
        
        # Check expected categories
        if "expected_categories" in case:
            detected = [c.name for c in result.all_categories[:len(case["expected_categories"])]]
            for expected_cat in case["expected_categories"]:
                if expected_cat not in detected:
                    case_passed = False
                    errors.append(f"Expected category '{expected_cat}' not found in {detected}")
        
        # Check expected sentiment
        if "expected_sentiment" in case:
            if result.sentiment != case["expected_sentiment"]:
                case_passed = False
                errors.append(f"Expected sentiment '{case['expected_sentiment']}', got '{result.sentiment}'")
        
        # Check expected urgency
        if "expected_urgency" in case:
            if result.urgency != case["expected_urgency"]:
                case_passed = False
                errors.append(f"Expected urgency '{case['expected_urgency']}', got '{result.urgency}'")
        
        if case_passed:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {case['description']}")
            print(f"    Text: '{text[:60]}...'")
            for error in errors:
                print(f"    Error: {error}")
    
    total = passed + failed
    print(f"\n  Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    return passed, failed, total


def test_additional_cases():
    """Test 100+ additional challenging cases."""
    print("\n" + "="*70)
    print("TEST: Additional Challenging Cases (100+)")
    print("="*70)
    
    passed = 0
    failed = 0
    errors = []
    
    for i, text in enumerate(ADDITIONAL_CASES):
        try:
            result = generate_recommendation(text)
            
            # Basic validation
            if not result.primary_category:
                failed += 1
                errors.append(f"Case {i+1}: No primary category detected")
                continue
            
            if not result.student_recommendation.summary:
                failed += 1
                errors.append(f"Case {i+1}: No student recommendation summary")
                continue
            
            if not result.admin_action_plan.investigation_steps:
                failed += 1
                errors.append(f"Case {i+1}: No admin investigation steps")
                continue
            
            if result.confidence < 0 or result.confidence > 1:
                failed += 1
                errors.append(f"Case {i+1}: Confidence out of range: {result.confidence}")
                continue
            
            passed += 1
            
        except Exception as e:
            failed += 1
            errors.append(f"Case {i+1}: Exception: {str(e)[:100]}")
    
    total = passed + failed
    print(f"\n  Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    if errors:
        print(f"\n  First 10 errors:")
        for error in errors[:10]:
            print(f"    {error}")
    
    return passed, failed, total


def test_student_admin_separation():
    """Test that student and admin recommendations are different."""
    print("\n" + "="*70)
    print("TEST: Student/Admin Recommendation Separation")
    print("="*70)
    
    test_texts = [
        "The wifi in the hostel has been down for two days",
        "My lecturer is always absent and does not teach",
        "The food in the canteen is terrible and overpriced",
        "I was robbed near the campus gate last night",
        "The library has no books for my course",
    ]
    
    passed = 0
    failed = 0
    
    for text in test_texts:
        result = generate_recommendation(text)
        
        student_text = result.student_recommendation.summary + " " + result.student_recommendation.immediate_action
        admin_text = " ".join(result.admin_action_plan.investigation_steps) + " " + " ".join(result.admin_action_plan.corrective_actions)
        
        # Check that they are not identical
        if student_text.strip() != admin_text.strip():
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: Student and admin recommendations are identical for: '{text[:50]}...'")
        
        # Check that student recommendation is simpler (shorter)
        if len(student_text) < len(admin_text) * 1.5:  # Allow some flexibility
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: Student recommendation too long for: '{text[:50]}...'")
    
    total = passed + failed
    print(f"\n  Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    return passed, failed, total


def test_fallback_behavior():
    """Test that unclear feedback gets a fallback response."""
    print("\n" + "="*70)
    print("TEST: Fallback Behavior for Unclear Feedback")
    print("="*70)
    
    unclear_cases = [
        "ok",
        "something",
        "help",
        "no",
        "yes",
        "maybe",
        "idk",
        "???",
    ]
    
    passed = 0
    failed = 0
    
    for text in unclear_cases:
        result = generate_recommendation(text)
        
        # Should have low confidence or fallback
        if result.confidence < 0.3 or result.fallback_used:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: '{text}' should have low confidence but got {result.confidence}")
    
    total = passed + failed
    print(f"\n  Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    return passed, failed, total


def test_multi_issue_detection():
    """Test that multi-issue feedback is detected."""
    print("\n" + "="*70)
    print("TEST: Multi-Issue Detection")
    print("="*70)
    
    multi_issue_cases = [
        ("The wifi is not working and the food is terrible", True),
        ("My lecturer is absent and the classroom has no lights", True),
        ("The bus is late and the driver is rude", True),
        ("The wifi is not working", False),
        ("The food is cold", False),
    ]
    
    passed = 0
    failed = 0
    
    for text, expected_multi in multi_issue_cases:
        result = generate_recommendation(text)
        
        if result.multi_issue == expected_multi:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: '{text[:60]}...'")
            print(f"    Expected multi_issue={expected_multi}, got {result.multi_issue}")
            print(f"    Categories: {[c.name for c in result.all_categories[:3]]}")
    
    total = passed + failed
    print(f"\n  Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    return passed, failed, total


def test_all_categories_covered():
    """Test that all 12 categories can be detected."""
    print("\n" + "="*70)
    print("TEST: All 12 Categories Coverage")
    print("="*70)
    
    detected_categories = set()
    
    for category, cases in CATEGORY_TEST_CASES.items():
        for case in cases[:3]:  # Test first 3 cases per category
            result = classify_categories(case)
            if result:
                detected_categories.add(result[0].name)
    
    expected = set(CATEGORY_DEFINITIONS.keys())
    missing = expected - detected_categories
    
    if not missing:
        print(f"  All {len(expected)} categories detected!")
        return 1, 0, 1
    else:
        print(f"  Missing categories: {missing}")
        return 0, 1, 1


def test_urgency_levels():
    """Test that urgency levels are correctly assigned."""
    print("\n" + "="*70)
    print("TEST: Urgency Level Assignment")
    print("="*70)
    
    test_cases = [
        ("There was a shooting near the campus", "critical"),
        ("A student was kidnapped at the gate", "critical"),
        ("I was robbed and injured last night", "high"),
        ("The wifi has been down for a week", "medium"),
        ("The library could use more books", "low"),
    ]
    
    passed = 0
    failed = 0
    
    for text, expected_urgency in test_cases:
        analysis = process_feedback(text)
        result = generate_recommendation(text, urgency_score=analysis['urgency_score'])
        
        if result.urgency == expected_urgency:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: '{text[:60]}...'")
            print(f"    Expected urgency={expected_urgency}, got {result.urgency}")
    
    total = passed + failed
    print(f"\n  Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    return passed, failed, total


# ==================== MAIN TEST RUNNER ====================

def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("COMPREHENSIVE RECOMMENDATION ENGINE TEST SUITE")
    print("="*70)
    
    all_results = []
    
    # Run all tests
    all_results.append(("Category Classification", test_category_classification()))
    all_results.append(("Sentiment Analysis", test_sentiment_analysis()))
    all_results.append(("Difficult Edge Cases", test_difficult_cases()))
    all_results.append(("Additional Cases (100+)", test_additional_cases()))
    all_results.append(("Student/Admin Separation", test_student_admin_separation()))
    all_results.append(("Fallback Behavior", test_fallback_behavior()))
    all_results.append(("Multi-Issue Detection", test_multi_issue_detection()))
    all_results.append(("All Categories Coverage", test_all_categories_covered()))
    all_results.append(("Urgency Levels", test_urgency_levels()))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total_passed = 0
    total_failed = 0
    
    for name, (passed, failed, total) in all_results:
        status = "PASS" if failed == 0 else "FAIL"
        print(f"  {status}: {name} - {passed}/{total} passed")
        total_passed += passed
        total_failed += failed
    
    grand_total = total_passed + total_failed
    pass_rate = total_passed / grand_total * 100 if grand_total > 0 else 0
    
    print(f"\n  TOTAL: {total_passed}/{grand_total} passed ({pass_rate:.1f}%)")
    
    if pass_rate >= 90:
        print("  STATUS: EXCELLENT")
    elif pass_rate >= 80:
        print("  STATUS: GOOD")
    elif pass_rate >= 70:
        print("  STATUS: ACCEPTABLE")
    else:
        print("  STATUS: NEEDS IMPROVEMENT")
    
    return pass_rate


if __name__ == "__main__":
    run_all_tests()
