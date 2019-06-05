import numpy as np

def person_distance(person_encodings, person_to_compare):
	if len(person_encodings) == 0:
		return np.empty((0))
	return np.linalg.norm(person_encodings - person_to_compare, axis=1)

def compare_persons(known_person_encodings, person_encoding_to_check, tolerance):
	print(person_distance(known_person_encodings, person_encoding_to_check))
	return list(person_distance(known_person_encodings, person_encoding_to_check) <= tolerance)

def person_recognizer(new_person_vector, known_person_encodings, known_person_names):
	matches = compare_persons(known_person_encodings, new_person_vector, tolerance=20)

	name = 'unknown_person'
	person_distances = person_distance(known_person_encodings, new_person_vector)
	best_match_index = np.argmin(person_distances)
	if matches[best_match_index]:
		name = known_person_names[best_match_index]

	print('linalg.norm the smallest distance result match:{}'.format(name))

	return known_person_names[best_match_index], name