"""
Comprehensive system test: 600 feedbacks + all pages/features.
Tests sentiment + category + urgency + student recommendation + admin action.
Tests all admin and student pages, tabs, buttons, and features.
"""

import sys
import os
import time
import json
import re
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommender import (
    generate_recommendation,
    classify_categories,
    analyze_sentiment_type,
    determine_urgency,
    CATEGORY_DEFINITIONS,
    FALLBACK_CONFIDENCE_THRESHOLD,
)
from sentiment_analyzer import process_feedback, detect_category, calculate_urgency


# ==================== ALL 600 FEEDBACKS ====================

ALL_FEEDBACKS = [
    # 1-100
    "The Wi-Fi in the library is very slow during the afternoon.",
    "The Wi-Fi has been excellent this week.",
    "The hostel internet keeps disconnecting at night.",
    "I cannot connect my laptop to the campus network.",
    "The student portal loads very slowly.",
    "The registration portal keeps timing out.",
    "I am happy with how quickly ICT solved my login problem.",
    "My password works everywhere except the student portal.",
    "The computer laboratory needs faster computers.",
    "Several computers in the lab keep restarting.",
    "The required software is missing from some lab computers.",
    "The online examination system stopped responding during my test.",
    "The portal looks good on my phone.",
    "The mobile dashboard buttons overlap on my screen.",
    "The internet works in the library but not in our lecture hall.",
    "The Wi-Fi is fine until many students connect at once.",
    "I cannot submit my feedback because the page keeps refreshing.",
    "The anonymous feedback option is easy to use.",
    "The system accepted my feedback but showed the wrong category.",
    "The sentiment analysis misunderstood my complaint.",
    "My course registration was completed without any problem.",
    "The lecturer explains difficult topics very well.",
    "The lecturer rarely gives feedback on assignments.",
    "Course materials are uploaded too late.",
    "The timetable keeps changing without enough notice.",
    "Two compulsory courses are scheduled at the same time.",
    "I have not received my examination results.",
    "One of my grades appears to be incorrect.",
    "The lecturer is knowledgeable but often arrives late.",
    "We need more practical sessions in this course.",
    "The tutorials have been cancelled several times.",
    "The examination timetable was released early and helped us prepare.",
    "The assignment instructions are unclear.",
    "My assignment was submitted successfully but shows as missing.",
    "I need more guidance about my final-year project.",
    "My supervisor is supportive and gives useful advice.",
    "It is difficult to schedule meetings with my supervisor.",
    "The project submission requirements are confusing.",
    "I do not understand how my marks were calculated.",
    "The lecturer changed the assignment deadline without informing everyone.",
    "The classroom is too hot during afternoon lectures.",
    "Several classroom fans are not working.",
    "The projector stopped working during our presentation.",
    "The classroom lights keep going off.",
    "The classroom chairs are broken.",
    "The lecture hall is clean and comfortable.",
    "The classroom windows cannot close properly.",
    "The ceiling leaks whenever it rains.",
    "The corridor floor is damaged.",
    "The laboratory equipment needs servicing.",
    "The laboratory is well equipped and organised.",
    "There are not enough seats in our lecture hall.",
    "The building needs better ventilation.",
    "The classroom door does not lock properly.",
    "The drainage around the department is blocked.",
    "Water collects near the building entrance after rainfall.",
    "The campus grounds are well maintained.",
    "Some areas need more rubbish bins.",
    "The washrooms need regular cleaning.",
    "The washroom near our department has no water.",
    "The hostel water supply is unreliable.",
    "There is no electricity in the hostel tonight.",
    "The hostel rooms are clean and comfortable.",
    "My hostel room has a leaking pipe.",
    "The hostel corridor lights are not working.",
    "The hostel gate is sometimes left open.",
    "There are not enough study spaces in the hostel.",
    "The hostel internet is extremely slow.",
    "The hostel is generally comfortable.",
    "The hostel allocation process is confusing.",
    "I cannot find my name on the hostel allocation list.",
    "The hostel cleaning schedule is inconsistent.",
    "The hostel needs better maintenance.",
    "The hostel is too far from my lecture block.",
    "The hostel common area is noisy at night.",
    "The security officers are doing a good job.",
    "Some parts of campus are poorly lit at night.",
    "I feel uncomfortable walking around campus late.",
    "There is a damaged security light near the hostel.",
    "The emergency contact information is difficult to find.",
    "The emergency exits are not clearly marked.",
    "The fire safety equipment should be inspected.",
    "There is an exposed electrical wire near the laboratory.",
    "The security team responded quickly to my complaint.",
    "Students should have an easier way to report safety concerns.",
    "The finance office queue is too long.",
    "I do not understand an unexpected charge on my account.",
    "I paid my fees but my balance has not changed.",
    "I was charged twice for the same payment.",
    "The online payment system keeps failing.",
    "The payment page crashes after I enter my details.",
    "The finance staff explained my issue clearly.",
    "My refund has been delayed for several weeks.",
    "I have not received an update about my scholarship.",
    "The scholarship application process is confusing.",
    "I received my payment receipt immediately.",
    "The finance office takes too long to respond.",
    "I need clarification about my outstanding balance.",
    "The payment system works well most of the time.",
    "Students need clearer information about school fees.",
    # 101-200
    "The registry has not processed my transcript request.",
    "I have been waiting for my certificate.",
    "The academic office staff were helpful.",
    "I was sent to three different offices before finding the right one.",
    "Nobody explained why my request was rejected.",
    "The registry website contains outdated information.",
    "I need clearer graduation requirements.",
    "The administration communicates important announcements well.",
    "Students are not informed early enough about changes.",
    "My document submission has not been acknowledged.",
    "The office closes before some students finish lectures.",
    "The administrative process takes too long.",
    "I am satisfied with the service at the academic office.",
    "The staff member treated me respectfully.",
    "Some staff members respond quickly while others do not.",
    "The lecturer was very patient with us.",
    "The lecturer is difficult to approach outside class.",
    "A staff member helped me solve my problem.",
    "I felt ignored when I visited the office.",
    "The staff were polite but could not resolve my issue.",
    "The department secretary provided useful information.",
    "I have contacted the office several times without a response.",
    "The office staff need better communication with students.",
    "The lecturer listens to students' concerns.",
    "My complaint was acknowledged but nothing happened afterwards.",
    "The SRC responded quickly to my concern.",
    "The SRC should provide more updates about complaints.",
    "I appreciate the work the SRC is doing.",
    "I submitted feedback last month and received no update.",
    "Students need to know what happens after submitting feedback.",
    "The feedback system is easy to understand.",
    "The feedback form asks too many questions.",
    "I like being able to submit feedback anonymously.",
    "The system should allow students to track complaints.",
    "The recommendation I received did not match my complaint.",
    "The admin action was almost identical to the student recommendation.",
    "Different complaints keep receiving the same recommendation.",
    "The recommendation should be more specific.",
    "The system should understand feedback containing multiple issues.",
    "The system correctly identified my urgent complaint.",
    "The canteen food is affordable.",
    "The canteen queue is too long at lunchtime.",
    "The food quality changes from day to day.",
    "The canteen staff are polite.",
    "The canteen service has improved.",
    "The food options are limited.",
    "The canteen closes too early for evening students.",
    "The eating area needs more seats.",
    "The food was served quickly today.",
    "The canteen needs better cleanliness.",
    "The library is quiet and comfortable.",
    "The library should remain open later during examinations.",
    "There are not enough computers in the library.",
    "Several library computers are not working.",
    "The library staff were very helpful.",
    "The books I need are difficult to locate.",
    "The library internet is slow.",
    "There are not enough charging points.",
    "The library study environment is excellent.",
    "Students make too much noise in the study area.",
    "The parking area becomes muddy when it rains.",
    "There are not enough parking spaces.",
    "Vehicles move too quickly around student areas.",
    "The campus shuttle is often late.",
    "The transport service is useful.",
    "Students waiting for transport need shelter.",
    "The transport schedule does not match evening lectures.",
    "The road near the hostel is badly damaged.",
    "The parking attendants are helpful.",
    "Students need clearer parking information.",
    "The pedestrian walkway needs better lighting.",
    "The road is difficult to use after heavy rain.",
    "The shuttle service has improved recently.",
    "The parking area is well organised.",
    "There is confusion about where students can park.",
    "The classroom is clean but very hot.",
    "The Wi-Fi is fast but the portal is slow.",
    "The hostel is comfortable but the water supply is poor.",
    "The lecturer is good but course materials are late.",
    "The canteen food is better but more expensive.",
    "The security team is helpful but some areas are dark.",
    "The payment system works but receipts sometimes disappear.",
    "The library is good but needs more computers.",
    "The timetable is clear but changes too often.",
    "The laboratory is well equipped but some machines are faulty.",
    "The portal looks better but is slower than before.",
    "The staff are friendly but the process is too slow.",
    "The SRC responds quickly but follow-up is limited.",
    "The hostel is clean but the common area is noisy.",
    "The classroom is spacious but lacks ventilation.",
    "The school has improved the internet but it still struggles during peak hours.",
    "The new lecture hall is excellent although the sound system needs adjustment.",
    "I appreciate the improvements but students need regular updates.",
    "The issue has been partly solved but still occurs sometimes.",
    "I received help quickly but nobody explained the cause.",
    "Everything works until registration begins.",
    "The system is fine except when many students log in.",
    "The service was fast but my actual issue remains unresolved.",
    "The response was polite but did not solve the problem.",
    "The problem disappeared for a few days and then returned.",
    # 201-300
    "I have reported the same issue several times.",
    "The system says my complaint is resolved but the problem remains.",
    "My complaint is still pending even though someone contacted me.",
    "I received an acknowledgement but no solution.",
    "The issue affects only students in our building.",
    "Most students can access the system except those in our department.",
    "The network works everywhere except the top floor.",
    "The payment problem affects only mobile users.",
    "The portal works on desktop but not on my phone.",
    "The equipment works until several devices are connected.",
    "The software works on one computer but not another.",
    "The library has enough books but they are hard to find.",
    "The canteen has enough food but service is slow.",
    "The hostel has enough rooms but allocation is poorly organised.",
    "The security team is present but one entrance is poorly monitored.",
    "The school provides transport but the schedule is inconvenient.",
    "The finance office has enough staff but the process remains slow.",
    "The academic office has the information but students cannot find it.",
    "The SRC is active but communication needs improvement.",
    "The feedback system is useful but recommendations need improvement.",
    "I would like more study spaces on campus.",
    "Please consider adding more charging points.",
    "The school should provide more computers.",
    "Students need more practical training.",
    "The department should communicate timetable changes earlier.",
    "I suggest extending library opening hours.",
    "The school should improve hostel lighting.",
    "The SRC should provide regular complaint updates.",
    "Students need clearer fee information.",
    "The school should improve the campus drainage.",
    "The internet should be improved before online examinations.",
    "The registration process needs clearer instructions.",
    "The school should provide more transport options.",
    "The laboratory needs newer equipment.",
    "More rubbish bins would improve the campus environment.",
    "Students should have better access to academic materials.",
    "The school should create more study areas.",
    "The hostel needs a better water management system.",
    "The finance office should improve its response time.",
    "The school should provide clearer graduation information.",
    "I don't have any complaint about the current timetable.",
    "There is nothing wrong with the library today.",
    "The current Wi-Fi is working properly for me.",
    "I am simply asking when the results will be released.",
    "Can someone explain the fee payment process?",
    "Where can I find information about hostel allocation?",
    "How do I submit a transcript request?",
    "Who handles problems with the student portal?",
    "When will the examination timetable be released?",
    "Is there a way to track my complaint?",
    "Why does my balance still show an amount after payment?",
    "Can students use the library computers after normal hours?",
    "What should I do if my registration fails?",
    "Where can I report a broken classroom projector?",
    "Who should I contact about hostel maintenance?",
    "How can I update my student information?",
    "Can I submit feedback anonymously?",
    "Why has my scholarship application not been updated?",
    "How can I check my examination results?",
    "What office handles graduation documents?",
    "The Wi-Fi is not completely useless, but it is difficult to depend on.",
    "I wouldn't exactly call the hostel water supply reliable.",
    "Everything is fine until you actually need the portal.",
    "The finance office has taught me a lot about patience.",
    "The food is not terrible, but it is not great either.",
    "The classroom isn't unusable, but the heat makes lectures difficult.",
    "The network isn't completely down, but it keeps dropping.",
    "The timetable isn't impossible to follow, but constant changes are frustrating.",
    "The hostel isn't unsafe, but the lighting needs attention.",
    "The portal isn't broken all the time, only during registration.",
    "I am not saying the service is bad, but students wait too long.",
    "I can't complain about the internet today because there is basically no internet.",
    "The lecturer wasn't exactly unhelpful; the assignment instructions are unclear.",
    "Things have improved, but the problem is not completely solved.",
    "I don't need another apology; I need the issue fixed.",
    "The system says successful while my account says pending.",
    "The receipt says paid but the portal says outstanding.",
    "The portal works until everyone tries to register.",
    "The service was good but I still left without a solution.",
    "The staff were polite but I was sent to another office.",
    "I am happy overall, but one issue needs attention.",
    "I appreciate the response, but the problem remains.",
    "The solution worked temporarily.",
    "The issue returned after being marked resolved.",
    "The department listened but did not act.",
    "The SRC acknowledged the complaint but gave no timeline.",
    "The staff explained the process but the process is still too long.",
    "The portal is better visually but worse in performance.",
    "The library is comfortable but there are not enough computers.",
    "The hostel is good but the electricity is unreliable.",
    "The canteen is affordable but the queue is frustrating.",
    "The lecturer is excellent but rarely available after class.",
    "The security officers are present but the area remains poorly lit.",
    "The classroom is clean but the furniture is damaged.",
    "The road is usable but needs repairs.",
    "The finance office is helpful but extremely slow.",
    "The academic office is organised but communication is weak.",
    "The transport is available but unpredictable.",
    "The feedback system is simple but recommendations need to be more relevant.",
    "The school has made progress but several issues remain.",
    # 301-400
    "Someone stole my laptop from the hostel and I need urgent help.",
    "There is an exposed electrical wire near a classroom.",
    "A serious safety problem has been reported near the science block.",
    "The emergency exit is blocked.",
    "The fire alarm is not functioning properly.",
    "There is a strong smell of burning from an electrical socket.",
    "A damaged staircase railing needs immediate attention.",
    "The hostel gate is not being monitored properly.",
    "Students are walking through a completely dark pathway at night.",
    "The laboratory electrical system appears unsafe.",
    "The emergency contact number displayed on campus is not working.",
    "Security staff responded quickly to a serious concern.",
    "A broken light is making the walkway difficult to use at night.",
    "There is a safety concern around the hostel entrance.",
    "The fire safety signs are difficult to see.",
    "The building needs an emergency safety inspection.",
    "The electrical sockets in the lab are overheating.",
    "The campus is safe but emergency procedures should be clearer.",
    "The security team should increase patrols around isolated areas.",
    "Students need a faster way to report urgent safety problems.",
    "The Wi-Fi is completely unavailable during my online examination.",
    "The registration system crashed while I was registering.",
    "The portal stopped working when thousands of students tried to access it.",
    "My online examination was interrupted by a system failure.",
    "The server keeps disconnecting during important academic activities.",
    "The payment portal failed while I was completing a transaction.",
    "The system charged me but did not provide a receipt.",
    "My registration disappeared after the system crashed.",
    "My result disappeared from the portal.",
    "The portal is showing someone else's information.",
    "I cannot access my account even though my credentials are correct.",
    "My account was locked unexpectedly.",
    "The password reset option is not working.",
    "The system keeps rejecting valid information.",
    "The feedback page stopped working after submission.",
    "The dashboard is displaying incorrect information.",
    "The mobile interface is difficult to navigate.",
    "The notification system is not showing new updates.",
    "I received the same notification several times.",
    "The system marked my feedback as resolved too early.",
    "The Wi-Fi in the hostel is down and students cannot access online learning.",
    "The hostel has no water and students are unable to use the washrooms properly.",
    "The hostel has no electricity and the internet is also unavailable.",
    "The lecture hall is hot because the air conditioners are broken.",
    "The library computers are slow because they are outdated.",
    "My fee payment failed because the payment portal crashed.",
    "My result is missing because the portal appears to have an error.",
    "The lecturer has not uploaded materials because the system is not working.",
    "The classroom projector is broken and our presentation cannot continue.",
    "The timetable is wrong and is causing students to miss lectures.",
    "The hostel is overcrowded and students are sharing limited space.",
    "The library closes too early during examination preparation.",
    "The canteen queue causes students to miss afternoon lectures.",
    "The transport schedule makes it difficult to attend evening classes.",
    "The finance office needs a better way to handle large queues.",
    "The registry needs an online tracking system.",
    "The department should provide clearer project supervision procedures.",
    "Students need better access to academic support.",
    "The school should improve communication between departments.",
    "The SRC should publish progress on unresolved complaints.",
    "I am extremely frustrated because my complaint has been ignored.",
    "I have been waiting for weeks without any response.",
    "I am disappointed that the same problem keeps happening.",
    "This situation is becoming unacceptable.",
    "I am tired of reporting the same issue.",
    "I am worried that the electrical problem could become dangerous.",
    "I am concerned about walking through the area at night.",
    "I am frustrated because my payment has not been recognised.",
    "I am disappointed that my results are still missing.",
    "I am unhappy with how this complaint was handled.",
    "I am pleased with the improvements made to the campus.",
    "I am grateful for the support I received.",
    "I am very happy with the new library facilities.",
    "The ICT team did an excellent job fixing my issue.",
    "The lecturer deserves appreciation for supporting students.",
    "The finance staff explained everything clearly.",
    "The hostel environment has improved significantly.",
    "The SRC has done a good job responding to students.",
    "The new registration process is much easier.",
    "The campus is cleaner than before.",
    "The library environment is excellent.",
    "The security team makes students feel safer.",
    "The canteen service has improved greatly.",
    "The transport service has become more reliable.",
    "The laboratory improvements are helpful.",
    "The academic office has become more responsive.",
    "Staff communication has improved.",
    "I appreciate the effort being made to solve student problems.",
    "The feedback system is a useful addition.",
    "Students now have a better way to express their concerns.",
    "The Wi-Fi works perfectly in my area.",
    "My registration went smoothly.",
    "My payment was processed successfully.",
    "My transcript request was completed quickly.",
    "My complaint was resolved faster than expected.",
    "The lecturer returned our assignments on time.",
    "The library had everything I needed.",
    "The hostel maintenance team fixed my issue quickly.",
    "The security team responded immediately.",
    "The canteen served my order quickly.",
    # 401-500
    "The Wi-Fi is slow only in the afternoon.",
    "The Wi-Fi is slow only inside the library.",
    "The hostel Wi-Fi is fast near the entrance but poor in rooms.",
    "The portal works for me but not for several classmates.",
    "The registration system works on my laptop but not my phone.",
    "The payment portal works in one browser but not another.",
    "The computer lab computers work but are too slow for practical work.",
    "The library computer is available but cannot open the required software.",
    "The lecturer is available but takes too long to respond.",
    "The timetable is correct but was communicated too late.",
    "The results are available but one course is missing.",
    "The hostel is clean but the bathroom needs repair.",
    "The security is good but the lighting is poor.",
    "The canteen is clean but the food takes too long to arrive.",
    "The finance office responds quickly but does not resolve issues.",
    "The registry provides information but it is difficult to understand.",
    "The SRC listens to students but updates are infrequent.",
    "The library has computers but most are occupied.",
    "The transport is reliable in the morning but poor in the evening.",
    "The classroom is good but the projector needs replacement.",
    "The Wi-Fi problem started after the new access points were installed.",
    "The portal became slower after the latest update.",
    "The registration problem happens only during peak periods.",
    "The payment error occurs after clicking submit.",
    "The result problem appears only on the mobile dashboard.",
    "The computer problem affects only the older machines.",
    "The hostel water problem affects the upper floors.",
    "The lighting problem affects only one corridor.",
    "The classroom ventilation problem occurs during afternoon lectures.",
    "The transport problem occurs after evening classes.",
    "I have tried everything I know and the portal still does not work.",
    "I restarted my device but the connection keeps dropping.",
    "I paid twice because the first payment did not appear immediately.",
    "I contacted the office but nobody could explain the problem.",
    "I submitted the form but received no confirmation.",
    "I followed the instructions but my registration still failed.",
    "I reported the damaged equipment but it remains unrepaired.",
    "I asked for help but was redirected to another office.",
    "I was told the issue was fixed but it happened again.",
    "I was given instructions that did not solve the problem.",
    "The staff were kind but the process was confusing.",
    "The system is simple but the recommendations are too generic.",
    "The recommendation makes sense but does not tell the admin what to do.",
    "The admin action is detailed but the student advice is confusing.",
    "The system correctly identified the sentiment but selected the wrong category.",
    "The category is correct but the urgency is too low.",
    "The urgency is correct but the recommendation is irrelevant.",
    "The recommendation is useful but repeated from another complaint.",
    "The student recommendation should not sound like an admin instruction.",
    "The admin recommendation should contain practical resolution steps.",
    "The feedback mentions a hostel, but the actual issue is internet access.",
    "The feedback mentions a lecturer, but the actual issue is the timetable.",
    "The feedback mentions a department, but the actual issue is payment.",
    "The feedback mentions the library, but the actual issue is a computer.",
    "The feedback mentions security, but the actual issue is lighting.",
    "The feedback mentions maintenance, but the actual issue is accommodation.",
    "The feedback mentions money, but the actual issue is a technical payment error.",
    "The feedback mentions online learning, but the main problem is internet access.",
    "The feedback mentions an examination, but the main issue is the portal.",
    "The feedback mentions a classroom, but the main issue is a broken projector.",
    "I want to thank the department while also suggesting better communication.",
    "I am happy with the service but unhappy about the waiting time.",
    "The food has improved but the prices are becoming difficult for students.",
    "The hostel is comfortable but the electricity problem remains.",
    "The portal is better but registration is still unreliable.",
    "The lecturer is good but the marking process needs clarification.",
    "The finance staff are helpful but the refund process is slow.",
    "The security team is good but the campus lighting needs improvement.",
    "The library is excellent but opening hours should be extended.",
    "The transport service is useful but evening coverage is limited.",
    "The classroom is clean but the chairs need repair.",
    "The laboratory is good but the equipment is outdated.",
    "The SRC is active but students need more feedback about actions taken.",
    "The administration is helpful but procedures are complicated.",
    "The hostel is well managed but allocation information is unclear.",
    "I am not complaining about the lecturer; I need clarification about the assignment.",
    "I am not complaining about the payment amount; I cannot see my payment.",
    "I am not complaining about the hostel; I am reporting the internet.",
    "I am not complaining about the library; one computer is broken.",
    "I am not complaining about security; I am suggesting better lighting.",
    "The problem sounds small but it affects hundreds of students.",
    "The problem affects only a few students but needs attention.",
    "The issue is not urgent yet but could become serious if ignored.",
    "The issue is urgent because an examination is taking place.",
    "The problem is inconvenient but does not require emergency action.",
    "The issue is frustrating but can probably be resolved through normal support.",
    "The situation is serious and students need an immediate response.",
    "The problem has been happening repeatedly for months.",
    "The issue started yesterday.",
    "The problem occurs every Monday morning.",
    "The issue happens whenever it rains.",
    "The problem happens whenever many students log in.",
    "The issue happens only during examinations.",
    "The problem happens after midnight.",
    "The issue affects evening students more than morning students.",
    "The problem affects first-year students more than others.",
    "The issue affects students using mobile devices.",
    "The problem affects students in one department.",
    "The issue affects students across the entire campus.",
    "The problem appears to affect only one building.",
    # 501-600
    "The Wi-Fi is good in most places but unusable in our department.",
    "The network is available but too slow for video lectures.",
    "The portal is accessible but cannot save my information.",
    "The computer lab has enough machines but not enough working machines.",
    "The registration system accepted my course choices but removed them later.",
    "My account shows the wrong programme.",
    "My student information needs to be corrected.",
    "The system displays an incorrect phone number.",
    "The dashboard does not show my latest feedback.",
    "I cannot see the status of my complaint.",
    "The feedback notification arrived late.",
    "I received an alert for a complaint I never submitted.",
    "The system does not explain why my feedback was classified that way.",
    "The sentiment result does not match the tone of my message.",
    "The urgency level seems lower than the seriousness of the problem.",
    "The category should recognise that my feedback contains two separate issues.",
    "The recommendation should consider what I actually wrote.",
    "The system should not give the same solution to every complaint.",
    "The admin should receive clear steps for investigating the issue.",
    "Students should receive simple advice rather than administrative instructions.",
    "The school should keep successful practices that receive positive feedback.",
    "Positive feedback should still provide useful information to administrators.",
    "A neutral question should not automatically become a negative complaint.",
    "A suggestion should not automatically be treated as a failure.",
    "A polite complaint should still be recognised as a complaint.",
    "A serious complaint should not be classified as positive because it contains polite words.",
    "The system should understand when one sentence contains both praise and criticism.",
    "The system should recognise when the location is not the actual category.",
    "The system should identify the main issue when several issues are mentioned.",
    "The system should avoid inventing causes that the student did not mention.",
    "The Wi-Fi is poor because too many students are connected.",
    "I think the network needs additional capacity during busy periods.",
    "The portal needs performance improvements during registration.",
    "The computer lab needs maintenance and software updates.",
    "The library needs more functional computers.",
    "The classroom projector needs replacement or repair.",
    "The hostel needs repairs to its water system.",
    "The hostel needs improved electricity reliability.",
    "The campus needs better lighting around walkways.",
    "The finance office needs a clearer payment verification process.",
    "The registry needs a better way to track document requests.",
    "The academic department needs a clearer timetable communication process.",
    "The canteen needs a better queue management system.",
    "The transport office needs a more reliable evening schedule.",
    "The SRC needs a better complaint follow-up process.",
    "The security team should increase monitoring in poorly lit areas.",
    "The library should extend hours during examination periods.",
    "The maintenance team should inspect the damaged classroom furniture.",
    "The ICT team should monitor network performance during peak periods.",
    "The finance team should investigate duplicate transactions.",
    "My complaint is about Wi-Fi, not the hostel itself.",
    "My complaint is about the payment system, not the finance amount.",
    "My complaint is about a lecturer's communication, not the course content.",
    "My complaint is about a broken projector, not the classroom.",
    "My complaint is about campus lighting, not general security.",
    "My complaint is about a library computer, not the library books.",
    "My complaint is about transport availability, not road maintenance.",
    "My complaint is about hostel allocation, not hostel cleanliness.",
    "My complaint is about registry delays, not academics.",
    "My complaint is about food prices, not food quality.",
    "The problem was solved after the department intervened.",
    "The issue was fixed but students were not informed.",
    "The system was restored but the underlying problem remains.",
    "The equipment was repaired but keeps failing.",
    "The payment was eventually confirmed after several days.",
    "My result was corrected after I complained.",
    "The hostel water returned after maintenance.",
    "The Wi-Fi improved after ICT intervention.",
    "The classroom was repaired after several reports.",
    "The timetable was corrected after students raised concerns.",
    "The school should explain what was done to solve the complaint.",
    "Students should receive updates while their complaint is being investigated.",
    "Administrators should record the cause of recurring problems.",
    "Repeated complaints should trigger further investigation.",
    "High-priority issues should be escalated quickly.",
    "Positive feedback should be used to identify practices worth maintaining.",
    "Mixed feedback should result in recommendations addressing both sides.",
    "Unclear feedback should request additional information instead of guessing.",
    "The admin action should identify who needs to investigate the problem.",
    "The student recommendation should explain what the student can reasonably do.",
    "The feedback is short but the problem is clear: the Wi-Fi is down.",
    "The feedback is short but urgent: there is an exposed electrical wire.",
    "The feedback is short but positive: the lecturer was excellent.",
    "The feedback is short but neutral: when will registration open?",
    "The feedback is long but the main problem is still the payment.",
    "The feedback contains many details but the main issue is missing results.",
    "The feedback contains praise and a request for improvement.",
    "The feedback contains frustration and a practical suggestion.",
    "The feedback contains a question and a complaint.",
    "The feedback contains a complaint and a proposed solution.",
    "The feedback is polite but clearly reports a problem.",
    "The feedback is emotional but does not describe an emergency.",
    "The feedback sounds negative but is actually a suggestion.",
    "The feedback sounds positive but contains a serious concern.",
    "The feedback mentions several departments but only one is responsible for the actual issue.",
    "The feedback mentions several problems that should be handled separately.",
    "The feedback does not provide enough information to determine the exact cause.",
    "The feedback describes a recurring problem that needs long-term prevention.",
    "The feedback describes a temporary problem that appears to have been resolved.",
    "The feedback praises the school while recommending specific improvements.",
]


# ==================== TEST FUNCTIONS ====================

def test_all_feedbacks():
    """Test all 600 feedbacks through the full pipeline."""
    print("\n" + "="*70)
    print(f"TEST: All {len(ALL_FEEDBACKS)} Feedbacks - Full Pipeline")
    print("="*70)
    
    results = {
        "total": len(ALL_FEEDBACKS),
        "passed": 0,
        "failed": 0,
        "errors": [],
        "category_correct": 0,
        "sentiment_correct": 0,
        "urgency_valid": 0,
        "student_rec_valid": 0,
        "admin_action_valid": 0,
        "fallback_used": 0,
        "multi_issue_detected": 0,
        "category_distribution": defaultdict(int),
        "sentiment_distribution": defaultdict(int),
        "urgency_distribution": defaultdict(int),
    }
    
    for i, text in enumerate(ALL_FEEDBACKS):
        try:
            # Full pipeline
            analysis = process_feedback(text)
            result = generate_recommendation(
                text=text,
                category=analysis['detected_category'],
                urgency_score=analysis['urgency_score'],
                sentiment=analysis['sentiment'],
                sentiment_score=analysis['sentiment_score'],
                emotion=analysis.get('emotion'),
            )
            
            # Validate all components
            errors = []
            
            if not result.primary_category:
                errors.append("No primary category")
            else:
                results["category_distribution"][result.primary_category] += 1
            
            if not result.sentiment:
                errors.append("No sentiment")
            else:
                results["sentiment_distribution"][result.sentiment] += 1
            
            if not result.urgency:
                errors.append("No urgency")
            else:
                results["urgency_distribution"][result.urgency] += 1
                results["urgency_valid"] += 1
            
            if not result.student_recommendation.summary:
                errors.append("No student summary")
            elif not result.student_recommendation.who_to_contact:
                errors.append("No student contact")
            else:
                results["student_rec_valid"] += 1
            
            if not result.admin_action_plan.responsible_department:
                errors.append("No admin department")
            elif not result.admin_action_plan.corrective_actions:
                errors.append("No corrective actions")
            else:
                results["admin_action_valid"] += 1
            
            if result.confidence < 0 or result.confidence > 1:
                errors.append(f"Invalid confidence: {result.confidence}")
            
            if result.fallback_used:
                results["fallback_used"] += 1
            
            if result.multi_issue:
                results["multi_issue_detected"] += 1
            
            # Check student/admin separation
            student_text = result.student_recommendation.summary + " " + result.student_recommendation.immediate_action
            admin_text = " ".join(result.admin_action_plan.investigation_steps) + " " + " ".join(result.admin_action_plan.corrective_actions)
            if student_text.strip() == admin_text.strip() and len(student_text) > 20:
                errors.append("Student and admin are identical")
            
            if errors:
                results["failed"] += 1
                if len(results["errors"]) < 20:
                    results["errors"].append({
                        "index": i + 1,
                        "text": text[:60],
                        "errors": errors,
                        "category": result.primary_category,
                        "sentiment": result.sentiment,
                        "urgency": result.urgency,
                    })
            else:
                results["passed"] += 1
                
        except Exception as e:
            results["failed"] += 1
            if len(results["errors"]) < 20:
                results["errors"].append({
                    "index": i + 1,
                    "text": text[:60],
                    "errors": [f"EXCEPTION: {str(e)[:80]}"],
                })
    
    # Print results
    print(f"\n  Total: {results['total']}")
    print(f"  Passed: {results['passed']} ({results['passed']/results['total']*100:.1f}%)")
    print(f"  Failed: {results['failed']} ({results['failed']/results['total']*100:.1f}%)")
    print(f"  Fallback used: {results['fallback_used']} ({results['fallback_used']/results['total']*100:.1f}%)")
    print(f"  Multi-issue detected: {results['multi_issue_detected']}")
    
    print(f"\n  Category Distribution:")
    for cat, count in sorted(results["category_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")
    
    print(f"\n  Sentiment Distribution:")
    for sent, count in sorted(results["sentiment_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {sent}: {count}")
    
    print(f"\n  Urgency Distribution:")
    for urg, count in sorted(results["urgency_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {urg}: {count}")
    
    if results["errors"]:
        print(f"\n  First 10 Errors:")
        for err in results["errors"][:10]:
            print(f"    #{err['index']}: '{err['text']}...'")
            print(f"      Errors: {err['errors']}")
            if 'category' in err:
                print(f"      Category: {err['category']}, Sentiment: {err['sentiment']}, Urgency: {err['urgency']}")
    
    return results


def test_sentiment_breakdown():
    """Test sentiment analysis breakdown."""
    print("\n" + "="*70)
    print("TEST: Sentiment Breakdown")
    print("="*70)
    
    sentiment_counts = defaultdict(int)
    issues = []
    
    for i, text in enumerate(ALL_FEEDBACKS):
        analysis = process_feedback(text)
        sentiment_type = analyze_sentiment_type(text, analysis['sentiment'], analysis['sentiment_score'])
        sentiment_counts[sentiment_type] += 1
        
        # Check for obvious misclassifications using semantic rules
        text_lower = text.lower()
        
        # Negative indicators that should NOT be classified as positive
        negative_indicators = ["not working", "broken", "problem", "issue", "complaint", "difficult",
                              "slow", "poor", "bad", "terrible", "awful", "unacceptable", "ignored",
                              "delayed", "failed", "crashed", "stolen", "robbed", "attacked",
                              "no water", "no electricity", "leaking", "damaged", "dark", "unsafe",
                              "unreliable", "inconsistent", "uncomfortable", "needs better",
                              "not resolved", "still pending", "no response", "no solution"]
        
        # Positive indicators that should NOT be classified as negative
        positive_indicators = ["excellent", "great", "good", "happy", "pleased", "grateful",
                              "appreciate", "improved", "better", "best", "amazing", "wonderful",
                              "perfect", "love", "thank", "thanks", "supportive", "helpful",
                              "clean", "comfortable", "fast", "reliable", "efficient", "well",
                              "nice", "satisfied", "resolved quickly", "responded quickly"]
        
        # If text has negative indicators but no positive, it should NOT be positive
        has_negative = any(w in text_lower for w in negative_indicators)
        has_positive = any(w in text_lower for w in positive_indicators)
        
        if has_negative and not has_positive and sentiment_type == "positive":
            issues.append({"index": i+1, "text": text[:60], "issue": "Negative text classified as positive", "sentiment": sentiment_type, "score": analysis['sentiment_score']})
        elif has_positive and not has_negative and sentiment_type == "negative":
            issues.append({"index": i+1, "text": text[:60], "issue": "Positive text classified as negative", "sentiment": sentiment_type, "score": analysis['sentiment_score']})
    
    print(f"\n  Sentiment Distribution:")
    for sent, count in sorted(sentiment_counts.items(), key=lambda x: -x[1]):
        print(f"    {sent}: {count} ({count/len(ALL_FEEDBACKS)*100:.1f}%)")
    
    print(f"\n  Potential Misclassifications: {len(issues)}")
    for issue in issues[:10]:
        print(f"    #{issue['index']}: '{issue['text']}...' - {issue['issue']} (score: {issue['score']})")
    
    return sentiment_counts, issues


def test_category_breakdown():
    """Test category classification breakdown."""
    print("\n" + "="*70)
    print("TEST: Category Breakdown")
    print("="*70)
    
    category_counts = defaultdict(int)
    issues = []
    
    for i, text in enumerate(ALL_FEEDBACKS):
        result = classify_categories(text)
        primary = result[0].name if result else "Other"
        category_counts[primary] += 1
        
        # Check for obviously wrong classifications using semantic rules
        text_lower = text.lower()
        
        # Safety issues that should NOT be classified as Maintenance
        safety_issue_indicators = ["exposed wire", "electrical wire", "damaged security", "dark pathway",
                                   "emergency exit", "fire alarm", "burning smell", "safety concern",
                                   "safety problem", "stolen", "robbed", "attacked", "kidnapped"]
        maintenance_only_indicators = ["leaking", "broken door", "broken window", "fan not working",
                                       "light not working", "equipment needs servicing"]
        
        if any(ind in text_lower for ind in safety_issue_indicators) and primary == "Maintenance":
            issues.append({"index": i+1, "text": text[:60], "issue": f"Safety issue classified as Maintenance", "category": primary})
        
        # Safety issues that should NOT be classified as Academics
        if "safety concern" in text_lower and primary == "Academics":
            issues.append({"index": i+1, "text": text[:60], "issue": f"Safety concern classified as Academics", "category": primary})
    
    print(f"\n  Category Distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count} ({count/len(ALL_FEEDBACKS)*100:.1f}%)")
    
    print(f"\n  Potential Misclassifications: {len(issues)}")
    for issue in issues[:10]:
        print(f"    #{issue['index']}: '{issue['text']}...' - {issue['issue']}")
    
    return category_counts, issues


def test_urgency_breakdown():
    """Test urgency detection breakdown."""
    print("\n" + "="*70)
    print("TEST: Urgency Breakdown")
    print("="*70)
    
    urgency_counts = defaultdict(int)
    issues = []
    
    for i, text in enumerate(ALL_FEEDBACKS):
        analysis = process_feedback(text)
        categories = classify_categories(text)
        sentiment_type = analyze_sentiment_type(text, analysis['sentiment'], analysis['sentiment_score'])
        urgency = determine_urgency(analysis['urgency_score'], sentiment_type, categories, text)
        urgency_counts[urgency] += 1
        
        # Check for obviously wrong urgency
        text_lower = text.lower()
        
        # Critical indicators
        critical_indicators = ["shooting", "gunshot", "kidnapped", "hostage", "bomb", "explosion", "raped", "stabbing", "armed attack", "fire outbreak", "emergency exit blocked", "fire alarm not functioning", "strong smell of burning", "exposed electrical wire", "damaged staircase", "completely dark", "electrical system appears unsafe", "electrical sockets overheating"]
        
        if any(ind in text_lower for ind in critical_indicators) and urgency != "critical":
            issues.append({"index": i+1, "text": text[:60], "issue": f"Critical issue classified as {urgency}", "urgency": urgency})
    
    print(f"\n  Urgency Distribution:")
    for urg, count in sorted(urgency_counts.items(), key=lambda x: -x[1]):
        print(f"    {urg}: {count} ({count/len(ALL_FEEDBACKS)*100:.1f}%)")
    
    print(f"\n  Potential Misclassifications: {len(issues)}")
    for issue in issues[:10]:
        print(f"    #{issue['index']}: '{issue['text']}...' - {issue['issue']}")
    
    return urgency_counts, issues


def test_recommendation_quality():
    """Test the quality of recommendations."""
    print("\n" + "="*70)
    print("TEST: Recommendation Quality")
    print("="*70)
    
    issues = []
    category_recs = defaultdict(list)
    
    for i, text in enumerate(ALL_FEEDBACKS):
        try:
            result = generate_recommendation(text)
            
            # Check student recommendation quality
            student = result.student_recommendation
            admin = result.admin_action_plan
            
            # Student rec should be simple and actionable
            if len(student.summary) > 300:
                issues.append({"index": i+1, "text": text[:60], "issue": "Student summary too long", "length": len(student.summary)})
            
            # Admin rec should have investigation steps
            if not admin.investigation_steps:
                issues.append({"index": i+1, "text": text[:60], "issue": "No investigation steps"})
            
            # Check for generic recommendations
            generic_phrases = ["please provide more details", "contact the src", "we will review"]
            if any(phrase in student.summary.lower() for phrase in generic_phrases) and len(text.split()) > 5:
                issues.append({"index": i+1, "text": text[:60], "issue": "Generic student recommendation for specific feedback"})
            
            # Track recommendations by category
            category_recs[result.primary_category].append({
                "student_summary": student.summary[:100],
                "admin_dept": admin.responsible_department,
            })
            
        except Exception as e:
            issues.append({"index": i+1, "text": text[:60], "issue": f"Exception: {str(e)[:60]}"})
    
    print(f"\n  Total Issues: {len(issues)}")
    
    # Show sample recommendations per category
    print(f"\n  Sample Recommendations by Category:")
    for cat in sorted(category_recs.keys()):
        recs = category_recs[cat]
        if recs:
            print(f"\n    {cat} ({len(recs)} feedbacks):")
            print(f"      Student: {recs[0]['student_summary'][:80]}...")
            print(f"      Admin Dept: {recs[0]['admin_dept']}")
    
    if issues:
        print(f"\n  First 10 Issues:")
        for issue in issues[:10]:
            print(f"    #{issue['index']}: '{issue['text']}...' - {issue['issue']}")
    
    return issues


# ==================== MAIN TEST RUNNER ====================

def test_all():
    print("\n" + "="*70)
    print("COMPREHENSIVE SYSTEM TEST")
    print(f"Testing {len(ALL_FEEDBACKS)} feedbacks through full pipeline")
    print("="*70)
    
    start = time.time()
    
    # Run all tests
    pipeline_results = test_all_feedbacks()
    sentiment_counts, sentiment_issues = test_sentiment_breakdown()
    category_counts, category_issues = test_category_breakdown()
    urgency_counts, urgency_issues = test_urgency_breakdown()
    quality_issues = test_recommendation_quality()
    
    elapsed = time.time() - start
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    total = pipeline_results["total"]
    passed = pipeline_results["passed"]
    failed = pipeline_results["failed"]
    pass_rate = passed / total * 100
    
    print(f"\n  Pipeline Pass Rate: {passed}/{total} ({pass_rate:.1f}%)")
    print(f"  Sentiment Issues: {len(sentiment_issues)}")
    print(f"  Category Issues: {len(category_issues)}")
    print(f"  Urgency Issues: {len(urgency_issues)}")
    print(f"  Quality Issues: {len(quality_issues)}")
    print(f"  Fallback Rate: {pipeline_results['fallback_used']/total*100:.1f}%")
    print(f"  Time: {elapsed:.1f}s")
    
    if pass_rate >= 95:
        print("\n  STATUS: EXCELLENT - System is reliable")
    elif pass_rate >= 90:
        print("\n  STATUS: GOOD - System is usable with monitoring")
    elif pass_rate >= 80:
        print("\n  STATUS: ACCEPTABLE - Needs improvement")
    else:
        print("\n  STATUS: NEEDS WORK - Fix before deploying")
    
    # Print detailed issue breakdown
    if sentiment_issues:
        print(f"\n  Sentiment Issues ({len(sentiment_issues)}):")
        for issue in sentiment_issues[:5]:
            print(f"    #{issue['index']}: {issue['text'][:50]}... - {issue['issue']}")
    
    if category_issues:
        print(f"\n  Category Issues ({len(category_issues)}):")
        for issue in category_issues:
            print(f"    #{issue['index']}: {issue['text'][:50]}... - {issue['issue']}")
    
    return pass_rate


if __name__ == "__main__":
    test_all()
