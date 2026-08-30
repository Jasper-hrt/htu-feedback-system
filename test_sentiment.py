from sentiment_analyzer import process_feedback

tests = [
    ('The Wi-Fi is working great!', 'Positive'),
    ('I love the new library', 'Positive'),
    ('The Wi-Fi is terrible', 'Negative'),
    ('I hate the food', 'Negative'),
    ('The staff are rude', 'Negative'),
    ('The building is on campus', 'Neutral'),
    ('Oh great, another power outage', 'Negative'),
    ('Thanks for the surprise fee', 'Negative'),
    ('The food is not terrible', 'Positive'),
    ('Could be worse', 'Positive'),
    ('Someone stole my laptop', 'Negative'),
    ('I have been waiting for my refund', 'Negative'),
    ('The queue is extremely long', 'Negative'),
    ('The timetable keeps changing', 'Negative'),
    ('There is no water', 'Negative'),
    ('The lights are flickering', 'Negative'),
]

correct = 0
for text, expected in tests:
    r = process_feedback(text)
    actual = r['sentiment']
    if actual == expected:
        correct += 1
    else:
        print(f'FAIL: {text[:45]} -> {actual} (expected {expected})')
print(f'Accuracy: {correct}/{len(tests)} ({correct/len(tests)*100:.1f}%)')
