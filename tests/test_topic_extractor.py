from sentiment.topic_extractor import extract_topics

def test_topics_detect_flooding_and_classroom():
    assert extract_topics('consistent flooding in the classroom')[:2] == ['Flooding / Drainage', 'Facilities / Maintenance']

def test_topics_detect_safety_context():
    assert 'Security / Safety' in extract_topics('security prevented a robbery')
