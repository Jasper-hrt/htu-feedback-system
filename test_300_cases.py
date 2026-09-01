"""
Comprehensive 300 test cases for the classification and recommendation engine.
Tests category accuracy, sentiment, urgency, multi-issue detection, and recommendation quality.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommender import (
    classify_categories,
    analyze_sentiment_type,
    generate_recommendation,
)
from sentiment_analyzer import process_feedback


# ==================== 300 TEST CASES ====================
# Each tuple: (text, expected_primary_category, expected_sentiment_type, expected_urgency, is_multi_issue)

TEST_CASES = [
    # ========== BLOCK 1: Core categories (1-96) ==========
    ("The Wi-Fi works fine in the morning but becomes almost unusable when everyone arrives for lectures.", "ICT", "mixed", "medium", False),
    ("I can access the portal, but every important page takes several minutes to load.", "ICT", "negative", "medium", False),
    ("The computer lab has enough computers, but many of them freeze when students run the required software.", "ICT", "mixed", "medium", False),
    ("My password works everywhere except on the student portal.", "ICT", "negative", "medium", False),
    ("The internet connection disappeared completely during our online class.", "ICT", "negative", "high", False),
    ("The ICT team fixed my problem quickly, and I am grateful for the support.", "ICT", "positive", "low", False),
    ("The registration portal is easy to use when it is not overloaded.", "ICT", "mixed", "medium", False),
    ("I keep getting disconnected from the network whenever I move between buildings.", "ICT", "negative", "medium", False),
    ("The lecturer uploaded the notes, but they are difficult to download because the network is slow.", "ICT", "mixed", "medium", False),
    ("The projector works sometimes and then suddenly stops during lectures.", "ICT", "negative", "medium", False),
    ("I have paid my fees but the portal still says I owe money.", "Finance", "negative", "high", False),
    ("My account shows two charges for one payment.", "Finance", "negative", "high", False),
    ("I don't understand why the amount on my student account keeps changing.", "Finance", "negative", "medium", False),
    ("The finance office answered my question, but I still don't know what the extra charge means.", "Finance", "mixed", "medium", False),
    ("I have submitted all my scholarship documents and have not received any update.", "Finance", "negative", "medium", False),
    ("The refund process seems to have stopped completely.", "Finance", "negative", "high", False),
    ("I paid yesterday and received a receipt, but my account has not been updated.", "Finance", "mixed", "high", False),
    ("The payment page crashes whenever I try to complete the transaction.", "Finance", "negative", "high", False),
    ("The finance staff were helpful and explained the payment process clearly.", "Finance", "positive", "low", False),
    ("I don't mind paying the fees, but students need clearer information about what they are paying for.", "Finance", "mixed", "low", False),
    ("The classroom ceiling is leaking whenever it rains.", "Maintenance", "negative", "medium", False),
    ("One of the classroom windows cannot close properly.", "Maintenance", "negative", "medium", False),
    ("The lights in the lecture hall keep going off during lessons.", "Maintenance", "negative", "medium", False),
    ("The corridor floor is damaged and students keep tripping over the broken section.", "Maintenance", "negative", "high", False),
    ("The classroom is clean, but the furniture needs attention.", "Maintenance", "mixed", "low", False),
    ("The washroom has been without water since yesterday.", "Maintenance", "negative", "medium", False),
    ("The toilet flushes are not working properly.", "Maintenance", "negative", "medium", False),
    ("The department building needs better ventilation.", "Maintenance", "negative", "medium", False),
    ("The classroom fans are noisy and barely provide enough air.", "Maintenance", "negative", "medium", False),
    ("The laboratory equipment keeps failing during practical sessions.", "Maintenance", "negative", "medium", False),
    ("The hostel is comfortable, but the water supply is unreliable.", "Accommodation", "mixed", "medium", False),
    ("The room is overcrowded and there is barely enough space for students.", "Accommodation", "negative", "medium", False),
    ("The hostel electricity goes off almost every evening.", "Accommodation", "negative", "medium", False),
    ("The hostel gate is sometimes left open without proper monitoring.", "Safety", "negative", "high", False),
    ("There is poor lighting between the hostel and lecture block.", "Safety", "negative", "medium", False),
    ("The hostel rooms are clean, but the common areas need attention.", "Accommodation", "mixed", "low", False),
    ("I reported a leaking pipe in my room and it has still not been repaired.", "Maintenance", "negative", "medium", False),
    ("The hostel is too far from my department and transportation is difficult.", "Accommodation", "negative", "low", False),
    ("There are not enough study spaces in the hostel.", "Accommodation", "negative", "low", False),
    ("The hostel internet is unreliable, especially at night.", "ICT", "negative", "medium", False),
    ("The security guard was helpful when I reported a problem.", "Safety", "positive", "low", False),
    ("I feel uncomfortable walking through the dark part of campus after evening lectures.", "Safety", "negative", "medium", False),
    ("There is a broken security light near the hostel entrance.", "Maintenance", "negative", "medium", False),
    ("Students should not have to worry about their belongings disappearing from shared spaces.", "Safety", "negative", "medium", False),
    ("The security officers respond quickly whenever students report concerns.", "Safety", "positive", "low", False),
    ("I saw someone trying to enter a restricted area without permission.", "Safety", "negative", "high", False),
    ("The emergency contact information is not easy to find.", "Administration", "negative", "medium", False),
    ("The fire safety equipment should be checked regularly.", "Safety", "negative", "medium", False),
    ("There is an exposed electrical wire near the classroom entrance.", "Maintenance", "negative", "high", False),
    ("The laboratory has an electrical problem that needs urgent inspection.", "Maintenance", "negative", "high", False),
    ("The lecturer explains everything well but rarely gives feedback on assignments.", "Academics", "mixed", "medium", False),
    ("I understand the lectures, but the course workload is becoming difficult to manage.", "Academics", "mixed", "medium", False),
    ("The course materials are always uploaded after we need them.", "Academics", "negative", "medium", False),
    ("The lecturer keeps changing the deadline without explaining why.", "Academics", "negative", "medium", False),
    ("I am satisfied with the teaching in this course.", "Academics", "positive", "low", False),
    ("The marking system for this course is not clear to students.", "Academics", "negative", "medium", False),
    ("My grade changed after I asked for a remark, and I don't understand why.", "Academics", "negative", "medium", False),
    ("The lecturer is knowledgeable but sometimes arrives very late.", "Academics", "mixed", "medium", False),
    ("We need more practical demonstrations instead of only theoretical explanations.", "Academics", "negative", "medium", False),
    ("The tutorials have been cancelled several times without replacement sessions.", "Academics", "negative", "medium", False),
    ("I have not received my semester results yet.", "Academics", "negative", "medium", False),
    ("One course is missing from my results even though I wrote the examination.", "Academics", "negative", "high", False),
    ("My results are visible, but one of the grades appears incorrect.", "Academics", "negative", "medium", False),
    ("The timetable gives us overlapping classes.", "Academics", "negative", "medium", False),
    ("Two compulsory courses have been scheduled at the same time.", "Academics", "negative", "medium", False),
    ("The timetable was changed after students had already made plans.", "Academics", "negative", "medium", False),
    ("The examination timetable was released too late for proper preparation.", "Academics", "negative", "medium", False),
    ("I appreciate that the department gives us enough notice about examinations.", "Academics", "positive", "low", False),
    ("I don't know who to contact about my missing transcript.", "Administration", "negative", "medium", False),
    ("My transcript request has been pending for weeks.", "Administration", "negative", "medium", False),
    ("I submitted a document to the registry and have received no confirmation.", "Administration", "negative", "medium", False),
    ("The registration process requires students to visit too many offices.", "Administration", "negative", "medium", False),
    ("The academic office staff were polite but could not resolve my issue.", "Administration", "mixed", "medium", False),
    ("I was sent from one office to another without anyone taking responsibility.", "Administration", "negative", "medium", False),
    ("The registry website contains information that appears outdated.", "Administration", "negative", "medium", False),
    ("The department communicates important changes clearly.", "Administration", "positive", "low", False),
    ("Nobody informed us that the submission deadline had changed.", "Administration", "negative", "medium", False),
    ("I need a clearer explanation of the requirements for graduation.", "Administration", "negative", "medium", False),
    ("The final-year project submission instructions are confusing.", "Academics", "negative", "medium", False),
    ("My supervisor is supportive, but meetings are difficult to schedule.", "Academics", "mixed", "medium", False),
    ("I cannot get feedback on my project chapters quickly enough.", "Academics", "negative", "medium", False),
    ("The project guidelines are useful and easy to understand.", "Academics", "positive", "low", False),
    ("I am not sure whether I have completed all the requirements for graduation.", "Administration", "negative", "medium", False),
    ("The department should provide more guidance to final-year students.", "Academics", "negative", "medium", False),
    ("The SRC has responded to my complaint, but the actual issue remains unresolved.", "Student Affairs", "mixed", "medium", False),
    ("I appreciate the SRC for listening to students.", "Student Affairs", "positive", "low", False),
    ("The SRC should communicate the progress of complaints more often.", "Student Affairs", "negative", "medium", False),
    ("I submitted feedback before and never received an update.", "Student Affairs", "negative", "medium", False),
    ("Students need an easier way to follow the status of their complaints.", "Student Affairs", "negative", "medium", False),
    ("The student representatives are approachable and willing to listen.", "Student Affairs", "positive", "low", False),
    ("I don't know which SRC office handles accommodation complaints.", "Student Affairs", "negative", "medium", False),
    ("The suggestion process is good, but students need to know what happens after submitting feedback.", "Student Affairs", "mixed", "medium", False),
    ("I like the new student feedback system.", "Student Affairs", "positive", "low", False),
    ("The feedback form is easy to complete but some of the questions are unclear.", "Student Affairs", "mixed", "medium", False),
    ("I submitted feedback anonymously because I was uncomfortable identifying myself.", "Student Affairs", "neutral", "low", False),
    ("I want to report an issue without being treated differently because I complained.", "Student Affairs", "negative", "medium", False),
    # ========== BLOCK 2: Catering, Library, Transport (97-150) ==========
    ("The canteen food has improved, but the waiting time is still too long.", "Catering", "mixed", "medium", False),
    ("The food is affordable but there are not enough options.", "Catering", "mixed", "medium", False),
    ("The canteen staff are polite but service becomes very slow at lunchtime.", "Catering", "mixed", "medium", False),
    ("The food quality is inconsistent from one day to another.", "Catering", "negative", "medium", False),
    ("The canteen closes before some evening students finish lectures.", "Catering", "negative", "medium", False),
    ("I appreciate the improvement in the cleanliness of the eating area.", "Catering", "positive", "low", False),
    ("The prices have increased and students need an explanation.", "Catering", "negative", "medium", False),
    ("There are not enough seats in the eating area during peak periods.", "Catering", "negative", "medium", False),
    ("The library is quiet and comfortable for studying.", "Library", "positive", "low", False),
    ("The library needs longer opening hours during examinations.", "Library", "negative", "medium", False),
    ("Most of the library computers are occupied whenever I need one.", "Library", "negative", "medium", False),
    ("Several library computers are not working.", "ICT", "negative", "medium", False),
    ("The library internet is slower than the internet elsewhere on campus.", "ICT", "negative", "medium", False),
    ("The books I need for my course are difficult to find.", "Library", "negative", "medium", False),
    ("The library staff were very helpful when I needed assistance.", "Library", "positive", "low", False),
    ("The study area becomes noisy during busy periods.", "Library", "negative", "medium", False),
    ("There are not enough charging points for students using laptops.", "Library", "negative", "medium", False),
    ("The parking area becomes difficult to use whenever it rains.", "Transport", "negative", "medium", False),
    ("There are not enough parking spaces near the lecture halls.", "Transport", "negative", "medium", False),
    ("Vehicles move too quickly around areas where students walk.", "Safety", "negative", "medium", False),
    ("The campus shuttle does not always arrive when expected.", "Transport", "negative", "medium", False),
    ("The transport service is useful, but the schedule needs improvement.", "Transport", "mixed", "medium", False),
    ("Students waiting for transport have no proper shelter.", "Transport", "negative", "medium", False),
    ("The road near the hostel is badly damaged.", "Maintenance", "negative", "medium", False),
    ("The pedestrian path needs better lighting.", "Safety", "negative", "medium", False),
    ("The parking attendants are helpful during busy periods.", "Transport", "positive", "low", False),
    ("There is confusion about where students are allowed to park.", "Transport", "negative", "medium", False),
    ("The school should provide clearer transport information.", "Transport", "negative", "medium", False),
    ("I was charged a fee that I believe I have already paid.", "Finance", "negative", "high", False),
    ("I don't know whether my outstanding balance is correct.", "Finance", "negative", "medium", False),
    ("The finance office queue is too long.", "Finance", "negative", "medium", False),
    ("I waited for hours before someone attended to my payment issue.", "Finance", "negative", "medium", False),
    ("The staff explained my financial problem clearly and respectfully.", "Finance", "positive", "low", False),
    ("I have tried contacting the finance office but have not received a response.", "Finance", "negative", "medium", False),
    ("The scholarship application information is difficult to understand.", "Finance", "negative", "medium", False),
    ("I submitted my scholarship application but cannot check its status.", "Finance", "negative", "medium", False),
    ("The refund was approved but the money has not reached my account.", "Finance", "negative", "high", False),
    ("The payment system is confusing for first-time users.", "Finance", "negative", "medium", False),
    ("The online payment page gives an error after I enter my details.", "Finance", "negative", "medium", False),
    ("The portal accepted my payment but generated no receipt.", "Finance", "negative", "medium", False),
    ("I received a receipt but the transaction does not appear in my account.", "Finance", "negative", "high", False),
    ("The finance office should provide clearer instructions for resolving payment problems.", "Finance", "negative", "medium", False),
    ("The staff member I spoke to was respectful, but the process itself took too long.", "Finance", "mixed", "medium", False),
    ("The department secretary helped me find the correct office.", "Administration", "positive", "low", False),
    ("The staff member refused to explain why my request was rejected.", "Staff", "negative", "medium", False),
    ("Some staff respond quickly while others take several days.", "Staff", "mixed", "medium", False),
    ("I am happy with the support I receive from my department.", "Staff", "positive", "low", False),
    ("The lecturer is excellent, but communication outside class is difficult.", "Academics", "mixed", "medium", False),
    ("The staff at the registry are usually helpful.", "Administration", "positive", "low", False),
    ("My complaint was acknowledged but nothing else happened.", "Administration", "negative", "medium", False),
    ("I have sent several messages and still don't know whether anyone is handling my request.", "Administration", "negative", "medium", False),
    ("The office closes before students who have afternoon lectures can get there.", "Administration", "negative", "medium", False),
    ("The staff were polite, but I had to return several times to complete one request.", "Administration", "mixed", "medium", False),
    ("The administration should make its procedures easier for students to understand.", "Administration", "negative", "medium", False),
    # ========== BLOCK 3: Accommodation, Safety, Maintenance (151-200) ==========
    ("The hostel has water, but the pressure is too low for students on the upper floors.", "Accommodation", "mixed", "medium", False),
    ("There is water in the hostel during the day but not when students return from lectures.", "Accommodation", "negative", "medium", False),
    ("The hostel electricity is available most of the time, but frequent outages are affecting online learning.", "Accommodation", "mixed", "medium", False),
    ("The room is fine except for a leaking ceiling.", "Maintenance", "mixed", "medium", False),
    ("The hostel is generally good, although maintenance requests take too long.", "Accommodation", "mixed", "medium", False),
    ("The hostel environment is quiet and suitable for studying.", "Accommodation", "positive", "low", False),
    ("There is a damaged staircase railing that should be repaired.", "Maintenance", "negative", "medium", False),
    ("The hostel corridor lights are not working properly.", "Maintenance", "negative", "medium", False),
    ("The hostel cleaning schedule is inconsistent.", "Accommodation", "negative", "medium", False),
    ("Students are leaving rubbish around the hostel because there are not enough bins.", "Maintenance", "negative", "medium", False),
    ("The hostel management responds quickly to maintenance requests.", "Accommodation", "positive", "low", False),
    ("The hostel room allocation process is confusing.", "Accommodation", "negative", "medium", False),
    ("I was allocated a room but cannot find my name on the hostel list.", "Accommodation", "negative", "medium", False),
    ("Students need clearer information about hostel rules.", "Accommodation", "negative", "medium", False),
    ("The hostel internet is excellent, but the electricity supply is unreliable.", "Accommodation", "mixed", "medium", False),
    ("There is a safety concern around the hostel because the path is poorly lit.", "Safety", "negative", "medium", False),
    ("The hostel gate closes too early for students with evening lectures.", "Accommodation", "negative", "medium", False),
    ("Security checks at the hostel are inconsistent.", "Safety", "negative", "medium", False),
    ("The campus feels safe during the day, but some areas become uncomfortable at night.", "Safety", "mixed", "medium", False),
    ("The security team handled a recent concern professionally.", "Safety", "positive", "low", False),
    ("Students should have an easy way to report safety concerns immediately.", "Safety", "negative", "medium", False),
    ("The emergency exits are not clearly marked.", "Safety", "negative", "medium", False),
    ("Some emergency signs are difficult to see.", "Safety", "negative", "medium", False),
    ("The fire alarm should be tested regularly.", "Safety", "negative", "medium", False),
    ("There is a damaged socket near the laboratory equipment.", "Maintenance", "negative", "medium", False),
    ("The electrical supply in the lab keeps fluctuating.", "Maintenance", "negative", "medium", False),
    ("The corridor becomes slippery when it rains.", "Maintenance", "negative", "medium", False),
    ("The roof leaks whenever there is heavy rain.", "Maintenance", "negative", "medium", False),
    ("The drainage around the building is blocked.", "Maintenance", "negative", "medium", False),
    ("Water collects around the entrance after rainfall.", "Maintenance", "negative", "medium", False),
    ("The building is clean but needs structural repairs.", "Maintenance", "mixed", "medium", False),
    ("The classroom door is broken and does not lock properly.", "Maintenance", "negative", "medium", False),
    ("Some desks are damaged but there are enough usable ones.", "Maintenance", "mixed", "low", False),
    ("The department needs more classrooms because classes are overcrowded.", "Maintenance", "negative", "medium", False),
    ("The lecture hall is large enough but ventilation is poor.", "Maintenance", "mixed", "medium", False),
    ("The laboratory is well equipped, but some equipment needs servicing.", "Maintenance", "mixed", "medium", False),
    ("Practical lessons are delayed because equipment is often unavailable.", "Academics", "negative", "medium", False),
    ("Students are not always informed when facilities are closed for maintenance.", "Maintenance", "negative", "medium", False),
    ("The maintenance team responded quickly to a broken fan.", "Maintenance", "positive", "low", False),
    ("I am impressed with the improvements made to the lecture halls.", "Maintenance", "positive", "low", False),
    ("The classroom is too cold in the morning but too hot later in the day.", "Maintenance", "negative", "medium", False),
    ("The lighting is good during the day but insufficient in the evening.", "Maintenance", "mixed", "medium", False),
    ("The building looks good, but some facilities are not functioning.", "Maintenance", "mixed", "medium", False),
    ("The new lecture hall is excellent, although the sound system needs adjustment.", "Maintenance", "mixed", "medium", False),
    ("The computer lab is clean and spacious.", "ICT", "positive", "low", False),
    ("The computers are working, but there are not enough for large classes.", "ICT", "mixed", "medium", False),
    ("The required software is missing from several computers.", "ICT", "negative", "medium", False),
    ("The lab computers restart unexpectedly.", "ICT", "negative", "medium", False),
    ("The software works on some computers but not others.", "ICT", "negative", "medium", False),
    ("Students need more technical support during practical sessions.", "ICT", "negative", "medium", False),
    # ========== BLOCK 4: Registration, Portal, Feedback (201-250) ==========
    ("I was able to register without any problems this semester.", "Administration", "positive", "low", False),
    ("Registration is easy until the system becomes busy.", "Administration", "mixed", "medium", False),
    ("The registration portal keeps timing out.", "ICT", "negative", "medium", False),
    ("My registered courses disappeared after I logged out.", "ICT", "negative", "medium", False),
    ("I registered for the wrong course because the available options were unclear.", "Administration", "negative", "medium", False),
    ("The system allowed me to select a course that I am not eligible to take.", "ICT", "negative", "medium", False),
    ("I cannot remove a course from my registration.", "ICT", "negative", "medium", False),
    ("The portal says my registration is complete, but my department says it is not.", "Administration", "negative", "medium", False),
    ("I received conflicting information from the portal and the academic office.", "Administration", "negative", "medium", False),
    ("The online registration system is much better than the old process.", "Administration", "positive", "low", False),
    ("The portal is working, but students need clearer instructions.", "ICT", "mixed", "medium", False),
    ("My student profile contains incorrect information.", "Administration", "negative", "medium", False),
    ("I cannot update my phone number on the portal.", "ICT", "negative", "medium", False),
    ("My account has been locked after several login attempts.", "ICT", "negative", "medium", False),
    ("The password reset process is confusing.", "ICT", "negative", "medium", False),
    ("I can log in from my phone but not from the computer lab.", "ICT", "negative", "medium", False),
    ("The portal works on some browsers but not others.", "ICT", "negative", "medium", False),
    ("The mobile version of the portal is difficult to use.", "ICT", "negative", "medium", False),
    ("Buttons overlap on my phone when I open the student dashboard.", "ICT", "negative", "medium", False),
    ("The dashboard loads properly on my laptop but looks broken on my phone.", "ICT", "negative", "medium", False),
    ("The feedback form works well on mobile.", "Student Affairs", "positive", "low", False),
    ("I cannot submit feedback because the form keeps refreshing.", "Student Affairs", "negative", "medium", False),
    ("The anonymous feedback option is easy to understand.", "Student Affairs", "positive", "low", False),
    ("I selected a category but the system changed it after submission.", "Student Affairs", "negative", "medium", False),
    ("The sentiment analysis seems to misunderstand what I am saying.", "Student Affairs", "negative", "medium", False),
    ("The system marked my complaint as positive even though I was reporting a problem.", "Student Affairs", "negative", "medium", False),
    ("The system correctly recognised that my feedback was urgent.", "Student Affairs", "positive", "low", False),
    ("The recommendation I received does not match my actual complaint.", "Student Affairs", "negative", "medium", False),
    ("The recommendation was useful, but the admin action was almost identical.", "Student Affairs", "negative", "medium", False),
    ("The system keeps giving me the same recommendation for different problems.", "Student Affairs", "negative", "medium", False),
    ("My feedback contains two issues, but the system only addresses one.", "Student Affairs", "negative", "medium", False),
    ("The system recognised the problem but suggested something the student cannot actually do.", "Student Affairs", "negative", "medium", False),
    ("The recommendation should be different for students and administrators.", "Student Affairs", "negative", "medium", False),
    ("The admin action should explain how to investigate the problem.", "Student Affairs", "negative", "medium", False),
    ("The system should provide a practical solution instead of repeating the complaint.", "Student Affairs", "negative", "medium", False),
    ("The feedback is positive, but the recommendation says to report a problem.", "Student Affairs", "negative", "medium", False),
    ("The feedback is neutral, but the system treats it as a serious complaint.", "Student Affairs", "negative", "medium", False),
    ("The system correctly identifies simple complaints but struggles with mixed feedback.", "Student Affairs", "mixed", "medium", False),
    ("The feedback mentions the hostel only as the location, but the actual problem is internet access.", "Student Affairs", "negative", "medium", False),
    ("The feedback mentions the library only as the location, but the issue is a broken computer.", "Student Affairs", "negative", "medium", False),
    ("The feedback mentions the department, but the actual issue is a payment problem.", "Student Affairs", "negative", "medium", False),
    ("The feedback mentions a lecturer, but the actual complaint concerns the timetable.", "Student Affairs", "negative", "medium", False),
    ("The feedback mentions security, but the actual issue is broken lighting.", "Student Affairs", "negative", "medium", False),
    ("The feedback mentions maintenance, but the actual issue is student accommodation.", "Student Affairs", "negative", "medium", False),
    ("The feedback mentions money, but the actual problem is a technical payment error.", "Student Affairs", "negative", "medium", False),
    ("The system should understand the difference between a suggestion and a complaint.", "Student Affairs", "negative", "medium", False),
    ("I am suggesting more study spaces, not reporting that the existing ones are broken.", "Student Affairs", "neutral", "low", False),
    ("I am asking for information about fees, not necessarily complaining about the fees.", "Student Affairs", "neutral", "low", False),
    ("I am praising the service while suggesting one improvement.", "Student Affairs", "mixed", "low", False),
    ("I am reporting a problem but I am not asking for punishment.", "Student Affairs", "negative", "medium", False),
    # ========== BLOCK 5: Mixed feedback, partial resolution, edge cases (251-300) ==========
    ("The school has improved the Wi-Fi, but it still struggles during peak hours.", "ICT", "mixed", "medium", False),
    ("The hostel is much cleaner now, although the water problem remains.", "Accommodation", "mixed", "medium", False),
    ("The lecturer is good, but the course organisation needs improvement.", "Academics", "mixed", "medium", False),
    ("The finance office is helpful, but the payment process is confusing.", "Finance", "mixed", "medium", False),
    ("The security team responds quickly, but the campus still has dark areas.", "Safety", "mixed", "medium", False),
    ("The library is excellent, but there are not enough computers.", "Library", "mixed", "medium", False),
    ("The canteen food is affordable, but service is slow.", "Catering", "mixed", "medium", False),
    ("The new portal looks better, but it is slower than the old one.", "ICT", "mixed", "medium", False),
    ("The classroom is comfortable, but the projector is unreliable.", "ICT", "mixed", "medium", False),
    ("The transport service is convenient, but the waiting time is unpredictable.", "Transport", "mixed", "medium", False),
    ("I appreciate the improvements, but students need regular updates.", "Administration", "mixed", "medium", False),
    ("The issue has been partly solved, but the original problem still occurs sometimes.", "Administration", "mixed", "medium", False),
    ("I was helped quickly, but I still don't understand what caused the problem.", "Administration", "mixed", "medium", False),
    ("Everything is working now, although it took several complaints to get here.", "Administration", "mixed", "medium", False),
    ("The staff were polite, but nobody explained the next step.", "Administration", "mixed", "medium", False),
    ("The system accepted my complaint, but I have no idea what happens next.", "Student Affairs", "mixed", "medium", False),
    ("I don't need another apology; I need the problem properly investigated.", "Administration", "negative", "medium", False),
    ("I am not saying the service is bad, but students should not have to wait this long.", "Administration", "mixed", "medium", False),
    ("The network isn't completely down, but it is unreliable enough to affect online classes.", "ICT", "mixed", "medium", False),
    ("The classroom isn't unusable, but the heat makes long lectures difficult.", "Maintenance", "mixed", "medium", False),
    ("The hostel isn't unsafe, but the lighting needs attention.", "Safety", "mixed", "medium", False),
    ("The food isn't terrible, but students deserve more consistent quality.", "Catering", "mixed", "medium", False),
    ("The portal isn't broken all the time, only when many students use it.", "ICT", "mixed", "medium", False),
    ("The timetable isn't impossible to follow, but constant changes are frustrating.", "Academics", "mixed", "medium", False),
    ("I am happy with the service overall, but one issue needs attention.", "Administration", "mixed", "medium", False),
    ("The department solved my problem but did not tell me why it happened.", "Administration", "mixed", "medium", False),
    ("The school responded to the complaint but did not provide a permanent solution.", "Administration", "mixed", "medium", False),
    ("The first response was helpful, but the follow-up has been poor.", "Administration", "mixed", "medium", False),
    ("The problem disappeared for a few days and then came back.", "Administration", "mixed", "medium", False),
    ("I was told the issue was resolved, but I experienced it again this morning.", "Administration", "negative", "medium", False),
    ("The system says my complaint is resolved even though nothing changed.", "Student Affairs", "negative", "medium", False),
    ("My complaint is still marked pending although someone contacted me about it.", "Student Affairs", "negative", "medium", False),
    ("I received an acknowledgement but no actual solution.", "Student Affairs", "negative", "medium", False),
    ("I have reported this problem multiple times and the same thing keeps happening.", "Student Affairs", "negative", "medium", False),
    ("The issue affects only some students, but it still needs attention.", "Student Affairs", "negative", "medium", False),
    ("Most students can access the system, but students in our building cannot.", "ICT", "negative", "medium", False),
    ("The network works everywhere except the top floor of the hostel.", "ICT", "negative", "medium", False),
    ("The payment problem affects only students using the mobile option.", "Finance", "negative", "medium", False),
    ("The portal works on desktop but not on mobile devices.", "ICT", "negative", "medium", False),
    ("The classroom equipment works until several devices are connected at once.", "ICT", "negative", "medium", False),
    ("The laboratory computers are fast, but the required software keeps crashing.", "ICT", "negative", "medium", False),
    ("The library has enough books, but students cannot easily locate them.", "Library", "mixed", "medium", False),
    ("The canteen has enough food, but the queue is too long.", "Catering", "mixed", "medium", False),
    ("The hostel has enough rooms, but allocation is poorly organised.", "Accommodation", "mixed", "medium", False),
    ("The security team is present, but some entrances are not monitored properly.", "Safety", "mixed", "medium", False),
    ("The school provides transport, but the schedule does not match evening lectures.", "Transport", "mixed", "medium", False),
    ("The finance office has enough staff, but the process is still slow.", "Finance", "mixed", "medium", False),
    ("The academic office has information available, but students cannot find it easily.", "Administration", "mixed", "medium", False),
    ("The SRC is active, but students need better communication about completed actions.", "Student Affairs", "mixed", "medium", False),
    ("The feedback system is useful, but recommendations should be more specific.", "Student Affairs", "mixed", "medium", False),
]


def run_tests():
    """Run all 300 test cases and report results."""
    print("\n" + "=" * 70)
    print("300 NEW TEST CASES - CLASSIFICATION & RECOMMENDATION ENGINE")
    print("=" * 70)

    total = len(TEST_CASES)
    cat_pass = 0
    cat_fail = 0
    sent_pass = 0
    sent_fail = 0
    urg_pass = 0
    urg_fail = 0
    multi_pass = 0
    multi_fail = 0
    rec_pass = 0
    rec_fail = 0

    cat_errors = []
    sent_errors = []
    urg_errors = []
    multi_errors = []
    rec_errors = []

    start_time = time.time()

    for i, (text, exp_cat, exp_sent, exp_urg, exp_multi) in enumerate(TEST_CASES, 1):
        # Test category classification
        cats = classify_categories(text)
        primary_cat = cats[0].name if cats else "Other"
        if primary_cat == exp_cat:
            cat_pass += 1
        else:
            cat_fail += 1
            if len(cat_errors) < 30:
                cat_errors.append(f"  {i}. '{text[:60]}...' Expected: {exp_cat}, Got: {primary_cat}")

        # Test sentiment and urgency via full recommendation
        try:
            result = generate_recommendation(text)
            actual_sent = result.sentiment
            actual_urg = result.urgency
            actual_multi = result.multi_issue
        except Exception as e:
            actual_sent = "error"
            actual_urg = "error"
            actual_multi = False

        if actual_sent == exp_sent:
            sent_pass += 1
        else:
            sent_fail += 1
            if len(sent_errors) < 20:
                sent_errors.append(f"  {i}. '{text[:60]}...' Expected: {exp_sent}, Got: {actual_sent}")

        if actual_urg == exp_urg:
            urg_pass += 1
        else:
            urg_fail += 1
            if len(urg_errors) < 20:
                urg_errors.append(f"  {i}. '{text[:60]}...' Expected: {exp_urg}, Got: {actual_urg}")

        if actual_multi == exp_multi:
            multi_pass += 1
        else:
            multi_fail += 1
            if len(multi_errors) < 20:
                multi_errors.append(f"  {i}. '{text[:60]}...' Expected multi={exp_multi}, Got: {actual_multi}")

        # Test recommendation quality (student vs admin separation)
        try:
            student_text = result.student_recommendation.summary + " " + result.student_recommendation.immediate_action
            admin_text = " ".join(result.admin_action_plan.investigation_steps[:2]) + " " + " ".join(result.admin_action_plan.corrective_actions[:2])
            if student_text.strip() != admin_text.strip() and len(student_text) < 500:
                rec_pass += 1
            else:
                rec_fail += 1
                if len(rec_errors) < 10:
                    rec_errors.append(f"  {i}. '{text[:60]}...' Student/Admin too similar or too long")
        except Exception:
            rec_fail += 1

    elapsed = time.time() - start_time

    # Print results
    print(f"\nTotal test cases: {total}")
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/total*1000:.0f}ms per case)")

    print(f"\n--- Category Classification ---")
    print(f"  Passed: {cat_pass}/{total} ({cat_pass/total*100:.1f}%)")
    print(f"  Failed: {cat_fail}/{total}")
    if cat_errors:
        print("  Sample failures:")
        for e in cat_errors[:15]:
            print(e)

    print(f"\n--- Sentiment Analysis ---")
    print(f"  Passed: {sent_pass}/{total} ({sent_pass/total*100:.1f}%)")
    print(f"  Failed: {sent_fail}/{total}")
    if sent_errors:
        print("  Sample failures:")
        for e in sent_errors[:10]:
            print(e)

    print(f"\n--- Urgency Assignment ---")
    print(f"  Passed: {urg_pass}/{total} ({urg_pass/total*100:.1f}%)")
    print(f"  Failed: {urg_fail}/{total}")
    if urg_errors:
        print("  Sample failures:")
        for e in urg_errors[:10]:
            print(e)

    print(f"\n--- Multi-Issue Detection ---")
    print(f"  Passed: {multi_pass}/{total} ({multi_pass/total*100:.1f}%)")
    print(f"  Failed: {multi_fail}/{total}")

    print(f"\n--- Recommendation Quality ---")
    print(f"  Passed: {rec_pass}/{total} ({rec_pass/total*100:.1f}%)")
    print(f"  Failed: {rec_fail}/{total}")

    total_pass = cat_pass + sent_pass + urg_pass + multi_pass + rec_pass
    total_tests = total * 5
    print(f"\n--- OVERALL ---")
    print(f"  Total checks: {total_tests}")
    print(f"  Passed: {total_pass} ({total_pass/total_tests*100:.1f}%)")
    print(f"  Failed: {total_tests - total_pass}")

    return total_pass, total_tests


if __name__ == "__main__":
    run_tests()
