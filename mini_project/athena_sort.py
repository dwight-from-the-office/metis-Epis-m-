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
    
    def __str__(self):
        output = "\nSchedule Overview:\n"
        for course, (room, time_slot) in self.assignments.items():
            output += f"{course.name} -> Room: {room.name}, Time Slot: {time_slot.time_id}\n"
        return output if self.assignments else "No schedule assigned yet"

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
            for time_slot in schedule.time_slots:
                if is_valid_assignment(schedule, course, room, time_slot):
                    domain.append((room, time_slot))

    return domain

def count_conflicts(schedule):
    conflicts = 0
    for course, (room, time_slot) in schedule.assignments.items():
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
    domains = {}

    for other_course in schedule.courses:
        if other_course != course and other_course not in schedule.assignments:
            domain = get_domain(other_course, schedule)
            domains[other_course] = domain.copy()

            if (room, time_slot) in domain:
                domain.remove((room,time_slot))

            if not domain:
                return False, domains
    return True, domains


# Backtrack Search
def backtracking_search(schedule: Schedule) -> bool:
    if len(schedule.assignments) == len(schedule.courses):
        return True # all courses are asigned
    
    course = select_unassigned_course_mrv(schedule)
    if course is None:
        return False
    
    domain = get_domain(course, schedule)

    for room , time_slot in domain:
        
        schedule.assignments[course] = (room, time_slot)

        is_valid, _ = forward_checking(schedule, course, room, time_slot)
        if is_valid:        
            if backtracking_search(schedule):
                return True
                

        del schedule.assignments[course]

                
           
    return False

def select_unassigned_course_mrv(schedule: Schedule) -> Optional[Course]:
    unassigned_courses = [c for c in schedule.courses if c not in schedule.assignments]

    # return course with the fewest valid assignments
    
    min_domain_size = float('inf')
    best_courses = []

    for course in unassigned_courses:
        domain_size = len(get_domain(course, schedule))
        if domain_size == 0:
            return None
        if domain_size < min_domain_size:
            min_domain_size = domain_size
            best_courses = [course]
        elif domain_size == min_domain_size:
            best_courses.append(course)

    if len(best_courses) == 1:
        return best_courses[0]

        
    def count_constraints(course: Course) -> int:
        # count of unsassigned courses
        count = 0
        for other_course in schedule.courses:
            if other_course != course and other_course not in schedule.assignments:
                if other_course.professor == course.professor:
                    count += 1
        return count
    
    return max(best_courses, key=count_constraints)

# Local Search
def min_conflicts(schedule, max_iterations):
    for course in schedule.courses:
        domain = get_domain(course, schedule)
        if domain:
            schedule.assignments[course] = random.choice(domain)
        else:
            return False
    for _ in range(max_iterations):
        conflicted_courses = find_conflicted_courses(schedule)

        if not conflicted_courses:
            return True  # no conflicts
        
        course = random.choice(conflicted_courses)
        domain = get_domain(course, schedule)

        if domain:
                schedule.assignments[course] = random.choice(domain)
        else:
            return False

    return False



def print_schedule_stats(schedule, method_name):
    print(f"\n=== {method_name} Results ===")
    for course, (room, time_slot) in schedule.assignments.items():
        print(f"{course.name} -> Room: {room.name}, Time Slot: {time_slot.time_id}")

    total_conflicts = count_conflicts(schedule)
    print(f"Conflicts: {total_conflicts}")
    print(f"Total Assignments: {len(schedule.assignments)} / {len(schedule.courses)}\n")

# Main function
def main():
    # used ai to populate data with test set starting here
        # Define 10 test courses
    courses = [
        Course("CS101", "Prof A", 30, [1, 2]),
        Course("CS102", "Prof B", 25, [2, 3]),
        Course("CS103", "Prof C", 40, [3, 4]),
        Course("CS104", "Prof D", 20, [1, 5]),
        Course("CS105", "Prof E", 35, [2, 4]),
        Course("CS106", "Prof A", 45, [3, 6]),
        Course("CS107", "Prof B", 50, [1, 3]),
        Course("CS108", "Prof C", 28, [2, 5]),
        Course("CS109", "Prof D", 32, [4, 6]),
        Course("CS110", "Prof E", 38, [3, 5])
    ]

    # Define 5 available rooms
    rooms = [
        Room("Room1", 50),
        Room("Room2", 40),
        Room("Room3", 30),
        Room("Room4", 35),
        Room("Room5", 45)
    ]

    # Define 6 time slots
    time_slots = [
        TimeSlot(1),
        TimeSlot(2),
        TimeSlot(3),
        TimeSlot(4),
        TimeSlot(5),
        TimeSlot(6)
    ]

    print(f"\nTesting with {len(courses)} courses, {len(rooms)} rooms, {len(time_slots)} time slots")

    # creates schedule
    schedule = Schedule(courses, rooms, time_slots)
    print(schedule)

    # backtracking search
    start_time = time.time()
    backtracking_success = backtracking_search(schedule)
    end_time = time.time()
    print(f"Execution Time (Backtracking): {end_time - start_time:.6f} seconds")


    # min-conflicts
    if backtracking_success:
        print_schedule_stats(schedule, "Backtracking")
    else:
        print("Backtracking Failed attempting min conflicts")


        schedule = Schedule(courses, rooms, time_slots)
        start = time.time()
        min_conflicts_success = min_conflicts(schedule, max_iterations=1000)
        end = time.time()


        print(f"Execution Time (Backtracking): {end - start:.4f} seconds")
        if min_conflicts_success:
            print_schedule_stats(schedule, "Min-Conflicts")
        else:
            print("Min-Conflicts also failed")



if __name__ == '__main__':
    main()


