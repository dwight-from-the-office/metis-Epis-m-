import time
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any, Optional


# Classes for Course, Room, TimeSlot, Schedule

class Course:
    def __init__(self, name: str, professor: str, size: int, perferred_times: Optional[List[int]] = None):
        self.name = name
        self.professor = professor
        self.size = size
        self.perferred_times = perferred_times
        self.assigned_slot = None

class Room:
    def __init__(self, name: str, capacity: int):
        self.name = name
        self.capacity = capacity

class TimeSlot:
    def __init__(self, time_id: int):
        self.time_id = time_id

class Schedule:
    def __init__(self, courses: List[Course], rooms: List[Room], time_slots: List[TimeSlot]):
        self.courses = courses
        self.rooms = rooms
        self.time_slots = time_slots
        self.assignments: Dict[Course, Tuple[Room, TimeSlot]] = {}

def is_valid_assignment(schedule, course, room, time_slot):
    # Check for capacity
    if room.capacity < course.size:
        return False
    
    # checks for conflicts with other courses
    for assigned_course, (assigned_room, assigned_time) in schedule.assignments.items():
        # professor cant teach two courses at the same time 
        if assigned_course.professor == course.professor and assigned_time == time_slot:
            return False
        # room cant be used by multiple courses at the same time
        if assigned_room == room and assigned_time == time_slot:
            return False
    
    return True

def get_domain(course, schedule):
    domain = []
    for room in schedule.rooms:
        # space avalable in room
        if room.capacity >= course.size:
            for time_slot in schedule.time_slot:
                if is_valid_assignment(schedule, course, room, time_slot):
                    domain.append((room, time_slot))

    return domain

def count_conflicts(schedule):
    conflicts = 0
    for course, (room, time_slot) in schedule.assignment.items():
        for other_course, (other_room, other_time) in schedule.assignments.items():
            if course != other_course:
                # Conflict professor assigned to two classes at the same time 
                if course.professor == other_course.professor and time_slot == other_time:
                    conflicts += 1
                # room has a course already
                if room == other_room and time_slot == other_time:
                    conflicts += 1
    
    return conflicts

def find_conflicted_courses(schedule):
    conflicted_courses = []
    for course in schedule.courses:
        if count_conflicts(schedule) > 0:
            conflicted_courses.append(course)
    return conflicted_courses




def forward_checking(schedule, course, room, time_slot):
    domains = defaultdict(list)

    for other_course in schedule.courses:
        if other_course != course:
            domain = get_domain(other_course, schedule)
            if (room, time_slot) in domain:
                domain.remove((room, time_slot)) 


# Backtrack Search
def backtracking_search(schedule: Schedule, course_index: int = 0) -> bool:
    if course_index >= len(schedule.courses):
        return True # all courses are asigned
    
    course = schedule.courses[course_index]
    for room in schedule.rooms:
        if room.capacitiy < course.size:
            continue # skips room if to many students in class for room


        for time_slot in schedule.time_slots:
            if is_valid_assignment(schedule, course, room, time_slot):
                schedule.assignments[course] = (room, time_slot)

                if backtracking_search(schedule, course_index + 1):
                    return True
                
                del schedule.assignments[course]
    
    return False